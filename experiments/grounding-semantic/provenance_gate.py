"""Canonical provenance gate for corpus admission (Round 13, author ruling 4).

The RAGBench arena parquet exposes no PMID / title / URL - only `id`
(e.g. `pubmedqa_11200`) and raw `documents` text - so the only executable
dedup between a candidate corpus and the arena is n-gram containment over
document text. Ruling 4 fixes the instrument: normalized 13-gram containment,
run BIDIRECTIONALLY, WARN at 0.5%, KILL at 2% of the candidate corpus.

Normalization: lowercase, punctuation to whitespace, whitespace collapsed.
Hashing: per-token blake2b-64, folded into a rolling polynomial over the
n-gram - deterministic across runs and processes, so results are reproducible.

Two modes:
  containment (default) - a unit "hits" if it shares ANY n-gram with the other
      side. Reported as the fraction of units carrying a hit, per direction,
      with the per-subset arena breakdown.
  --jaccard T - a unit "hits" if its n-gram set reaches Jaccard >= T against
      ANY single unit on the other side (ruling 2 uses n=8, T=0.3).

Torch-free, CPU-only, Polars only.

Run:
  uv run python experiments/grounding-semantic/provenance_gate.py \
      --candidate data/external/datasets/wice/claim_train.jsonl \
      --text-col claim --n 13 --arena-subsets hagrid,hotpotqa \
      --label wice_claims --out /tmp/gate.json
"""

import argparse
import hashlib
import io
import json
import pathlib
import re
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
ARCHIVE = ROOT / "data" / "external" / "datasets" / "dataset-ragbench.zip"

# Arena construction - byte-identical to R8-H77_unseen_arena.load_subsets()
# (that module is not imported: it pins CUDA env at import time and this gate
# is CPU-only).
MAX_CHUNKS = 8
N_PER_SUBSET = 250

WARN_DEFAULT = 0.005
KILL_DEFAULT = 0.02

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_MULT = np.uint64(1099511628211)  # FNV prime, used as the polynomial base


def normalize(text: str) -> str:
    """lowercase -> punctuation to space -> collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


class _TokenHasher:
    """Deterministic token -> uint64, memoized across the whole run."""

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        c = self.cache
        out = np.empty(len(tokens), dtype=np.uint64)
        for i, t in enumerate(tokens):
            v = c.get(t)
            if v is None:
                v = int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "big")
                c[t] = v
            out[i] = v
        return out


def ngram_hashes(text, n, hasher):
    """uint64 hashes of every n-gram of the normalized text (empty if too short)."""
    toks = normalize(text).split()
    if len(toks) < n:
        return np.empty(0, dtype=np.uint64)
    ids = hasher(toks)
    m = len(ids) - n + 1
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(n):
            h = h * _MULT + ids[k : k + m]
    return np.unique(h)


class _Side:
    """One side of the comparison: units grouped into named buckets."""

    def __init__(self, name):
        self.name = name
        self.buckets = {}  # bucket -> list of uint64 arrays (one per unit)

    def add(self, bucket, hashes):
        self.buckets.setdefault(bucket, []).append(hashes)

    @property
    def n_units(self):
        return sum(len(v) for v in self.buckets.values())

    @property
    def n_scorable(self):
        return sum(1 for v in self.buckets.values() for a in v if a.size)

    def index(self):
        """Per-bucket (sorted hashes, owning unit ids) plus a global sorted set."""
        idx = {}
        for b, units in self.buckets.items():
            if not units:
                continue
            flat = np.concatenate(units) if units else np.empty(0, dtype=np.uint64)
            owner = np.concatenate(
                [np.full(a.size, i, dtype=np.int64) for i, a in enumerate(units)]
            ) if units else np.empty(0, dtype=np.int64)
            order = np.argsort(flat, kind="stable")
            idx[b] = (flat[order], owner[order], len(units))
        return idx


def _hit_mask(query, sorted_hashes):
    """True where any query n-gram is present in the sorted hash array."""
    if query.size == 0 or sorted_hashes.size == 0:
        return False
    lo = np.searchsorted(sorted_hashes, query, side="left")
    hi = np.searchsorted(sorted_hashes, query, side="right")
    return bool(np.any(hi > lo))


def _max_jaccard(query, sorted_hashes, owner, unit_sizes):
    """Best Jaccard of the query n-gram set against any single indexed unit."""
    if query.size == 0 or sorted_hashes.size == 0:
        return 0.0, -1
    lo = np.searchsorted(sorted_hashes, query, side="left")
    hi = np.searchsorted(sorted_hashes, query, side="right")
    nz = np.nonzero(hi > lo)[0]
    if nz.size == 0:
        return 0.0, -1
    ids = np.concatenate([owner[lo[i] : hi[i]] for i in nz])
    uids, inter = np.unique(ids, return_counts=True)
    union = query.size + unit_sizes[uids] - inter
    j = inter / np.maximum(union, 1)
    k = int(np.argmax(j))
    return float(j[k]), int(uids[k])


def load_arena(subsets=None):
    """Arena document chunks, keyed by subset. Mirrors the H77 arena sample."""
    z = zipfile.ZipFile(ARCHIVE)
    side = _Side("arena")
    raw = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        if subsets and sub not in subsets:
            continue
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            continue
        df = df.sample(min(N_PER_SUBSET, len(df)), seed=0)
        raw[sub] = [c for d in df["documents"].to_list() for c in d[:MAX_CHUNKS]]
    return raw, side


def load_candidate(path, text_col):
    """Candidate text units from a parquet or a jsonl file."""
    p = pathlib.Path(path)
    if p.suffix == ".parquet":
        df = pl.read_parquet(p)
    else:
        df = pl.read_ndjson(p)
    if text_col not in df.columns:
        raise SystemExit(f"--text-col {text_col!r} not in {df.columns}")
    col = df[text_col]
    if col.dtype == pl.List(pl.String):
        return [" ".join(x) for x in col.to_list() if x is not None]
    return [t for t in col.to_list() if t]


def run_gate(
    candidate_texts,
    n=13,
    arena_subsets=None,
    jaccard=None,
    warn=WARN_DEFAULT,
    kill=KILL_DEFAULT,
    label="candidate",
    arena_texts=None,
):
    """Bidirectional gate. Returns the result dict written to JSON."""
    hasher = _TokenHasher()
    if arena_texts is None:
        arena_texts, _ = load_arena(arena_subsets)

    arena = _Side("arena")
    for sub, chunks in arena_texts.items():
        for c in chunks:
            arena.add(sub, ngram_hashes(c, n, hasher))
    cand = _Side(label)
    for t in candidate_texts:
        cand.add(label, ngram_hashes(t, n, hasher))

    a_idx = arena.index()
    c_idx = cand.index()
    a_sizes = {b: np.array([u.size for u in arena.buckets[b]], dtype=np.int64) for b in a_idx}
    c_sizes = {b: np.array([u.size for u in cand.buckets[b]], dtype=np.int64) for b in c_idx}

    def direction(src, dst_idx, dst_sizes):
        """Fraction of src units hitting each dst bucket, and any bucket."""
        units = [u for v in src.buckets.values() for u in v]
        per_bucket = {}
        any_hit = np.zeros(len(units), dtype=bool)
        best = np.zeros(len(units))
        detail = [None] * len(units)
        for b, (h, owner, _) in dst_idx.items():
            hits = np.zeros(len(units), dtype=bool)
            for i, q in enumerate(units):
                if jaccard is None:
                    hits[i] = _hit_mask(q, h)
                else:
                    j, uid = _max_jaccard(q, h, owner, dst_sizes[b])
                    best[i] = max(best[i], j)
                    hits[i] = j >= jaccard
                    if hits[i] and detail[i] is None:
                        detail[i] = {"bucket": b, "unit": uid, "jaccard": round(j, 4)}
            per_bucket[b] = {
                "units_with_hit": int(hits.sum()),
                "fraction": round(float(hits.mean()) if len(units) else 0.0, 6),
            }
            any_hit |= hits
        out = {
            "n_units": len(units),
            "units_with_hit": int(any_hit.sum()),
            "fraction": round(float(any_hit.mean()) if len(units) else 0.0, 6),
            "per_arena_subset": per_bucket,
        }
        if jaccard is not None and len(units):
            # how far below the bar the corpus actually sits
            out["best_jaccard"] = {
                "max": round(float(best.max()), 4),
                "p99": round(float(np.percentile(best, 99)), 4),
                "mean": round(float(best.mean()), 4),
            }
        return out, any_hit, detail

    fwd, fwd_hits, fwd_detail = direction(cand, a_idx, a_sizes)
    rev, _, _ = direction(arena, c_idx, c_sizes)

    worst = max(fwd["fraction"], rev["fraction"])
    verdict = "KILL" if worst >= kill else ("WARN" if worst >= warn else "PASS")

    examples = [
        {"candidate_unit": int(i), **(fwd_detail[i] or {})}
        for i in np.nonzero(fwd_hits)[0][:10].tolist()
    ]

    return {
        "instrument": "provenance_gate.py (R13 ruling 4)",
        "mode": "jaccard" if jaccard is not None else "containment",
        "n": n,
        "jaccard_threshold": jaccard,
        "thresholds": {"warn": warn, "kill": kill},
        "candidate": {
            "label": label,
            "n_units": cand.n_units,
            "n_units_scorable": cand.n_scorable,
        },
        "arena": {
            "subsets": {b: len(v) for b, v in arena.buckets.items()},
            "n_units": arena.n_units,
            "n_units_scorable": arena.n_scorable,
        },
        "candidate_vs_arena": fwd,
        "arena_vs_candidate": rev,
        "max_fraction": round(worst, 6),
        "verdict": verdict,
        "hit_examples": examples,
    }


def spike_control(candidate_texts, arena_texts, n=13, jaccard=None, k=10, label="spike"):
    """Positive control: inject k arena units into the candidate side and check
    they are all detected. Guards against a gate that cannot fire - the exact
    defect ruling 4 was issued to repair."""
    injected = [c for chunks in arena_texts.values() for c in chunks[: max(1, k // len(arena_texts))]][:k]
    res = run_gate(
        list(candidate_texts) + injected,
        n=n,
        arena_texts=arena_texts,
        jaccard=jaccard,
        label=label,
    )
    base = len(candidate_texts)
    hit = res["candidate_vs_arena"]["units_with_hit"]
    return {"injected": len(injected), "detected_total": hit, "baseline_hits": max(hit - len(injected), 0), "passes": hit >= len(injected), "candidate_units": base}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate", required=True, help="parquet or jsonl of candidate units")
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--n", type=int, default=13)
    ap.add_argument("--arena-subsets", default=None, help="comma list; default all 10")
    ap.add_argument("--jaccard", type=float, default=None, help="Jaccard mode at this threshold")
    ap.add_argument("--warn", type=float, default=WARN_DEFAULT)
    ap.add_argument("--kill", type=float, default=KILL_DEFAULT)
    ap.add_argument("--label", default="candidate")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    subs = [s.strip() for s in a.arena_subsets.split(",")] if a.arena_subsets else None
    texts = load_candidate(a.candidate, a.text_col)
    res = run_gate(
        texts,
        n=a.n,
        arena_subsets=subs,
        jaccard=a.jaccard,
        warn=a.warn,
        kill=a.kill,
        label=a.label,
    )
    res["candidate"]["path"] = str(a.candidate)
    res["candidate"]["text_col"] = a.text_col
    out = json.dumps(res, indent=2)
    print(out)
    if a.out:
        pathlib.Path(a.out).write_text(out + "\n")


if __name__ == "__main__":
    main()
