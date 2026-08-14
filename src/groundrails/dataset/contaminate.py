"""Contamination stage - is a candidate corpus already inside a walled corpus?

The instrument is n-gram overlap over normalized document text, run
BIDIRECTIONALLY: candidate against walled, and walled against candidate. A
candidate that shares documents with an evaluation corpus inflates every later
number, and the direction that catches it depends on which side is the subset -
so both are measured and the worse fraction rules.

Two modes. Containment (``jaccard=None``) counts a unit as hit if it shares ANY
n-gram with the other side; Jaccard mode counts it as hit only if its n-gram set
reaches the threshold against a SINGLE unit on the other side, which is the
form used for admission (8-gram, Jaccard >= 0.3).

Defaults WARN at 0.5% and KILL at 2% of the candidate corpus.

A gate that cannot fire is worse than no gate, so :func:`spike_control` injects
known walled units into the candidate side and requires every one of them back.
That control is why the verdict is trustworthy; run it before reading the gate.

Normalization is lowercase, punctuation to whitespace, whitespace collapsed.
Hashing is per-token blake2b-64 folded into a rolling polynomial, so results are
identical across runs, processes and machines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
import re

import numpy as np

WARN_DEFAULT = 0.005
KILL_DEFAULT = 0.02
N_DEFAULT = 8
JACCARD_DEFAULT = 0.3

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS = re.compile(r"\s+")
_MULT = np.uint64(1099511628211)  # FNV prime, used as the polynomial base


def normalize(text: str) -> str:
    """lowercase -> punctuation to space -> collapse whitespace."""
    return _WS.sub(" ", _PUNCT.sub(" ", text.lower())).strip()


class _TokenHasher:
    """Deterministic token -> uint64, memoized across the whole run."""

    def __init__(self):
        self.cache: dict[str, int] = {}

    def __call__(self, tokens):
        c = self.cache
        out = np.empty(len(tokens), dtype=np.uint64)
        for i, t in enumerate(tokens):
            v = c.get(t)
            if v is None:
                v = int.from_bytes(
                    hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "big"
                )
                c[t] = v
            out[i] = v
        return out


def ngram_hashes(text: str, n: int, hasher: _TokenHasher):
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

    def __init__(self, name: str):
        self.name = name
        self.buckets: dict[str, list] = {}

    def add(self, bucket: str, hashes) -> None:
        self.buckets.setdefault(bucket, []).append(hashes)

    @property
    def n_units(self) -> int:
        return sum(len(v) for v in self.buckets.values())

    @property
    def n_scorable(self) -> int:
        return sum(1 for v in self.buckets.values() for a in v if a.size)

    def index(self) -> dict:
        """Per-bucket (sorted hashes, owning unit ids, unit count)."""
        idx = {}
        for b, units in self.buckets.items():
            if not units:
                continue
            flat = np.concatenate(units)
            owner = np.concatenate(
                [np.full(a.size, i, dtype=np.int64) for i, a in enumerate(units)]
            )
            order = np.argsort(flat, kind="stable")
            idx[b] = (flat[order], owner[order], len(units))
        return idx


def _hit_mask(query, sorted_hashes) -> bool:
    """True where any query n-gram is present in the sorted hash array."""
    if query.size == 0 or sorted_hashes.size == 0:
        return False
    lo = np.searchsorted(sorted_hashes, query, side="left")
    hi = np.searchsorted(sorted_hashes, query, side="right")
    return bool(np.any(hi > lo))


def _max_jaccard(query, sorted_hashes, owner, unit_sizes) -> tuple[float, int]:
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


def gate(
    candidate_texts: Sequence[str],
    walled_texts: Mapping[str, Sequence[str]],
    *,
    n: int = N_DEFAULT,
    jaccard: float | None = JACCARD_DEFAULT,
    warn: float = WARN_DEFAULT,
    kill: float = KILL_DEFAULT,
    label: str = "candidate",
) -> dict:
    """Run the bidirectional gate and return the result record.

    ``walled_texts`` maps a bucket name (one walled corpus, or one of its
    subsets) to its document texts; the per-bucket breakdown says WHICH walled
    corpus a hit came from, which is what a quarantine decision needs.
    """
    hasher = _TokenHasher()
    walled = _Side("walled")
    for bucket, chunks in walled_texts.items():
        for c in chunks:
            walled.add(bucket, ngram_hashes(c, n, hasher))
    cand = _Side(label)
    for t in candidate_texts:
        cand.add(label, ngram_hashes(t, n, hasher))

    w_idx, c_idx = walled.index(), cand.index()
    w_sizes = {b: np.array([u.size for u in walled.buckets[b]], dtype=np.int64) for b in w_idx}
    c_sizes = {b: np.array([u.size for u in cand.buckets[b]], dtype=np.int64) for b in c_idx}

    def direction(src: _Side, dst_idx: dict, dst_sizes: dict):
        """Fraction of src units hitting each dst bucket, and any bucket."""
        units = [u for v in src.buckets.values() for u in v]
        per_bucket = {}
        any_hit = np.zeros(len(units), dtype=bool)
        best = np.zeros(len(units))
        detail: list = [None] * len(units)
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
            "per_walled_bucket": per_bucket,
        }
        if jaccard is not None and len(units):
            # how far below the bar the corpus actually sits
            out["best_jaccard"] = {
                "max": round(float(best.max()), 4),
                "p99": round(float(np.percentile(best, 99)), 4),
                "mean": round(float(best.mean()), 4),
            }
        return out, any_hit, detail

    fwd, fwd_hits, fwd_detail = direction(cand, w_idx, w_sizes)
    rev, _, _ = direction(walled, c_idx, c_sizes)

    worst = max(fwd["fraction"], rev["fraction"])
    verdict = "KILL" if worst >= kill else ("WARN" if worst >= warn else "PASS")
    return {
        "mode": "jaccard" if jaccard is not None else "containment",
        "n": n,
        "jaccard_threshold": jaccard,
        "thresholds": {"warn": warn, "kill": kill},
        "candidate": {
            "label": label,
            "n_units": cand.n_units,
            "n_units_scorable": cand.n_scorable,
        },
        "walled": {
            "buckets": {b: len(v) for b, v in walled.buckets.items()},
            "n_units": walled.n_units,
            "n_units_scorable": walled.n_scorable,
        },
        "candidate_vs_walled": fwd,
        "walled_vs_candidate": rev,
        "max_fraction": round(worst, 6),
        "verdict": verdict,
        "hit_examples": [
            {"candidate_unit": int(i), **(fwd_detail[i] or {})}
            for i in np.nonzero(fwd_hits)[0][:10].tolist()
        ],
    }


def spike_control(
    candidate_texts: Sequence[str],
    walled_texts: Mapping[str, Sequence[str]],
    *,
    n: int = N_DEFAULT,
    jaccard: float | None = JACCARD_DEFAULT,
    k: int = 10,
    label: str = "spike",
) -> dict:
    """Positive control: inject k walled units and require every one back.

    Guards against a gate that cannot fire - the failure mode where a clean
    verdict means the instrument is broken rather than the corpus clean.

    ``baseline_hits`` is the candidate's OWN hits, on top of the injected units.
    A non-zero baseline is not a control failure: it is contamination, and the
    gate's warn/kill fractions are what rule on its size. ``baseline_clean``
    reports it separately so the two are never confused.
    """
    per_bucket = max(1, k // max(len(walled_texts), 1))
    injected = [c for chunks in walled_texts.values() for c in list(chunks)[:per_bucket]][:k]
    res = gate(
        list(candidate_texts) + injected,
        walled_texts,
        n=n,
        jaccard=jaccard,
        label=label,
    )
    hit = res["candidate_vs_walled"]["units_with_hit"]
    baseline = max(hit - len(injected), 0)
    return {
        "injected": len(injected),
        "detected_total": hit,
        "baseline_hits": baseline,
        "baseline_clean": baseline == 0,
        "candidate_units": len(candidate_texts),
        "passes": hit >= len(injected) and len(injected) > 0,
    }


def check(
    candidate_texts: Sequence[str],
    walled_texts: Mapping[str, Sequence[str]],
    *,
    n: int = N_DEFAULT,
    jaccard: float | None = JACCARD_DEFAULT,
    warn: float = WARN_DEFAULT,
    kill: float = KILL_DEFAULT,
    label: str = "candidate",
    spike_k: int = 10,
    spike_sample: int = 2000,
) -> dict:
    """The stage: spike control first, then the gate, then one status.

    GREEN means the control fired and the corpus is under the kill bar; RED
    means it is not admissible. The spike control runs over a sample of the
    candidate (``spike_sample``) because it only has to prove the instrument
    fires, not measure the corpus.
    """
    spike = spike_control(
        list(candidate_texts)[:spike_sample],
        walled_texts,
        n=n,
        jaccard=jaccard,
        k=spike_k,
        label=f"{label}_spike",
    )
    result = gate(
        candidate_texts,
        walled_texts,
        n=n,
        jaccard=jaccard,
        warn=warn,
        kill=kill,
        label=label,
    )
    ok = spike["passes"] and result["verdict"] != "KILL"
    return {
        "corpus": label,
        "instrument": (
            f"{n}-gram "
            + (f"Jaccard >= {jaccard}" if jaccard is not None else "containment")
            + f", bidirectional, WARN {warn} / KILL {kill}"
        ),
        "spike_control": spike,
        "gate": result,
        "status": "GREEN" if ok else "RED",
    }


def walled_texts_from_files(
    paths: Sequence[Path | str],
    text_col: str = "chunk",
) -> dict[str, list[str]]:
    """Load walled corpora from files, one bucket per file (named by its stem).

    Reads parquet / jsonl / json via polars, or a text file as one unit per
    non-empty line. A list-of-strings column is joined per row, which is how a
    documents column carrying several passages becomes one unit.
    """
    from groundrails.dataset._deps import polars

    pl = polars()
    out: dict[str, list[str]] = {}
    for raw in paths:
        p = Path(raw)
        if p.suffix in (".txt", ".text"):
            out[p.stem] = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            continue
        if p.suffix == ".parquet":
            df = pl.read_parquet(p)
        elif p.suffix in (".jsonl", ".ndjson"):
            df = pl.read_ndjson(p)
        elif p.suffix == ".json":
            df = pl.read_json(p)
        else:
            raise ValueError(f"unsupported walled-corpus file: {p.name}")
        if text_col not in df.columns:
            raise ValueError(f"{p.name} has no column {text_col!r}; has {df.columns}")
        col = df[text_col]
        if col.dtype == pl.List(pl.String):
            out[p.stem] = [" ".join(x) for x in col.to_list() if x]
        else:
            out[p.stem] = [t for t in col.to_list() if t]
    return out
