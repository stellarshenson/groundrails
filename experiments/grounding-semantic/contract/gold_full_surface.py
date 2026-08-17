"""Contract verification of the `gold_full` evaluation surface.

`gold_full` is the campaign's in-domain held-out surface: 2,752 claims loaded by
`R10-H108_lane.gold_full()` from `private-rag-forensics/R7-H51_teacher_pairs.parquet`.
It carries the in-domain hold (bar >= 0.84).

This script does FOUR things and adjudicates none of them.

  1. RE-VERIFY, not trust, the prior split audit (`R20_goldfull_split_audit.py`,
     `logs/R20_goldfull_split_audit.log`).  Every headline number is recomputed
     from scratch through the same banked instruments and compared, value by
     value, against the banked artifact.  A mismatch is the finding.
  2. EXTEND to the clauses that audit did not cover - C2 in all three string
     forms in BOTH directions on claims, evidence, pairs AND documents; the
     DOCUMENT channel by 8-gram containment; and C6's eval-facing key lookup.
  3. C3 - measure, from the artifacts, what axis `gold_full` is actually split
     on relative to the training mix.
  4. Bank `contract/gold_full_surface_report.json`.

PRIVACY.  This surface is private data.  No verbatim claim, evidence or document
text - and no client or company name - is written to any artifact, log or
return.  Counts, rates, and blake2b fingerprints only.  Every text object stays
in memory.

MIX AMBIGUITY, measured both ways.  The prior audit assembled the mix from
`R18-H150_arm_run.LANES` (2 lanes, 14 DANN groups, 721,210 rows).  The phase-1
contract reports assemble it from `R20-H174_arm_run.LANES` (5 lanes, 17 groups,
760,618 rows).  The 14-group mix is a strict tag-subset of the 17-group mix, so
both are measured from one assembly and both are reported.

CPU ONLY - `CUDA_VISIBLE_DEVICES` is forced empty before any import.
HF_HUB_OFFLINE=1.  Polars, never pandas.  sklearn for the probe metric only.

Run:  nohup setsid uv run python \
        experiments/grounding-semantic/contract/gold_full_surface.py \
        2>&1 | tee logs/gold_full_surface.log &
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import hashlib
import importlib.util
import io
import json
import pathlib
import random
import re
import time
import zipfile

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent           # .../contract
SEM = HERE.parent                              # .../grounding-semantic
ROOT = SEM.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "gold_full_surface_report.json"
BANKED_AUDIT = SEM / "R20_goldfull_split_audit.json"

GOLD_PAIRS = SEM / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD_SRC = SEM / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"
VITC = DATA / "dataset-vitaminc.zip"

EXTRA_TAGS = ("frame_reject", "attr_pool", "path_bind")  # the 3 H174-only lanes

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, SEM / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading banked modules (CPU, CUDA_VISIBLE_DEVICES='')")
AUD = _mod("goldaudit", "R20_goldfull_split_audit.py")   # the audit being re-verified
ARM = AUD.ARM
H150 = AUD.H150
H174 = _mod("h174", "R20-H174_arm_run.py")
G = AUD.G                                                # provenance_gate
SERVE_CHARS = AUD.SERVE_CHARS

_WS_RE = re.compile(r"\s+")


# --- privacy-safe fingerprints -------------------------------------------------


def _h(s):
    return hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest()


def fingerprint(strings):
    """Order-independent blake2b fingerprint of a set of strings. No text leaks."""
    h = hashlib.blake2b(digest_size=16)
    for d in sorted(_h(s) for s in set(strings)):
        h.update(d)
    return h.hexdigest()


# --- string forms (C2) ---------------------------------------------------------


def f_raw(s):
    return s


def f_trunc(s):
    return s[:SERVE_CHARS]


def f_ws(s):
    return _WS_RE.sub(" ", s).strip().casefold()


FORMS = {"raw": f_raw, "truncated_1500": f_trunc, "ws_collapsed_casefold": f_ws}
FORM_DEFS = {
    "raw": "the string as loaded",
    "truncated_1500": f"s[:CFG.chunk_max_chars] ({SERVE_CHARS})",
    "ws_collapsed_casefold": "re.sub(r'\\s+', ' ', s).strip().casefold()",
}


# --- assembly ------------------------------------------------------------------


def assemble_mix17():
    """The 17-group portfolio mix (760,618). The 14-group mix is a tag-subset."""
    with ARM.untruncated_evidence():
        claims, chunks, y, tags = AUD.H108.public_train()
    log(f"clean public mix: {len(y)} rows (expected {H174.EXPECTED_CLEAN_ROWS})")
    if len(y) != H174.EXPECTED_CLEAN_ROWS:
        raise SystemExit("ABORT: clean mix row count is not the incumbent's")
    for fname, group, n_rows, n_pairs, fams in H174.LANES:
        df = pl.read_parquet(SEM / fname)
        got_fams = {r["neg_family"]: int(r["count"]) for r in df["neg_family"].value_counts().to_dicts()}
        got_pairs = df["pair_id"].n_unique()
        if len(df) != n_rows or got_pairs != n_pairs or got_fams != fams:
            raise SystemExit(f"LANE ABORT ({group}): {len(df)} rows / {got_pairs} pairs {got_fams}")
        claims += df["claim"].to_list()
        chunks += df["chunk"].to_list()
        tags += [group] * len(df)
        log(f"lane {group}: {len(df)} rows, {got_pairs} pairs")
    mix = pl.DataFrame({"claim": claims, "chunk": chunks, "tag": tags})
    names = tuple(sorted(set(tags)))
    if names != H174.EXPECTED_GROUPS or mix.height != H174.EXPECTED_MIX_ROWS:
        raise SystemExit(f"ASSEMBLY ABORT: {mix.height} rows, {len(names)} groups")
    log(f"assembled mix17: {mix.height} rows, {len(names)} groups")
    return mix


def assemble_gold():
    """Two views of gold_full, cross-checked against each other.

    `gold`  - byte-identical to the audit's construction: the flattened output of
              `R10-H108_lane.gold_full()`, whose owner index is the POSITION in
              that loader's output, not the archive's `owner` value.
    `goldp` - the pair parquet itself, carrying the true `owner`, which is what
              joins to the document parquet. Used for the document channel and C6.
    """
    cl, ck, y = AUD.H108.gold_full()
    rows = {"owner_ix": [], "claim": [], "chunk": []}
    for i, (c, ks) in enumerate(zip(cl, ck, strict=True)):
        for k in ks:
            rows["owner_ix"].append(i)
            rows["claim"].append(c)
            rows["chunk"].append(k[:SERVE_CHARS])
    gold = pl.DataFrame(rows)
    goldp = (pl.read_parquet(GOLD_PAIRS)
             .select(["owner", "claim", "chunk", "label"])
             .with_columns(pl.col("chunk").str.slice(0, SERVE_CHARS))
             .rename({"owner": "owner_ix"}))
    src = pl.read_parquet(GOLD_SRC).with_row_index("owner_ix")
    log(f"gold_full: {len(y)} claims, {gold.height} pairs, "
        f"{gold['claim'].n_unique()} unique claims, {gold['chunk'].n_unique()} unique chunks, "
        f"{src['source_text'].n_unique()} unique documents")
    return gold, goldp, src, len(y), y


def owner_alignment(gold, goldp, src):
    """Do the three gold views describe the same rows, and does `owner` join to
    the document parquet's row index?"""
    a = gold.select(["claim", "chunk"]).sort(["claim", "chunk"])
    b = goldp.select(["claim", "chunk"]).sort(["claim", "chunk"])
    same_pairs = a.equals(b)
    per = (goldp.group_by("owner_ix")
           .agg(pl.col("claim").first().alias("c"),
                pl.col("label").first().alias("l"),
                pl.col("claim").n_unique().alias("nc"))
           .join(src.select(["owner_ix", "claim", "label", "lang"])
                 .with_columns(pl.col("owner_ix").cast(pl.Int64)), on="owner_ix", how="left"))
    ok_claim = int((per["c"] == per["claim"]).sum())
    ok_label = int((per["l"] == per["label"]).sum())
    out = {
        "loader_view_and_parquet_view_have_identical_pair_multiset": bool(same_pairs),
        "owners": int(per.height),
        "owners_with_one_claim_string": int((per["nc"] == 1).sum()),
        "claim_string_matches_document_parquet_row": ok_claim,
        "label_matches_document_parquet_row": ok_label,
        "aligned": bool(ok_claim == per.height and ok_label == per.height),
    }
    log(f"owner alignment: pair multiset identical={same_pairs}; "
        f"claim {ok_claim}/{per.height}, label {ok_label}/{per.height} -> aligned={out['aligned']}")
    return out


# --- generic collision channel (C2) --------------------------------------------


def collide(label, gold_units, mix_units, n_gold_claims):
    """One (channel, form) cell, BOTH directions, with per-tag attribution.

    gold_units - DataFrame(owner_ix, u)   one row per gold (owner, unit)
    mix_units  - DataFrame(u, tag)        one row per mix ROW (not deduplicated)
    """
    g = gold_units.unique()
    m_rows = mix_units
    mix_rows_total = m_rows.height
    mix14_rows_total = m_rows.filter(~pl.col("tag").is_in(list(EXTRA_TAGS))).height
    m_uni = m_rows.unique()
    hits = g.join(m_uni, on="u", how="inner")           # gold unit x contributing tag
    n_hit_units = int(hits["u"].n_unique())
    n_hit_owners = int(hits["owner_ix"].n_unique())
    per_tag = {r["tag"]: int(r["len"]) for r in hits.group_by("tag").len().sort("len", descending=True).iter_rows(named=True)}
    h14 = hits.filter(~pl.col("tag").is_in(list(EXTRA_TAGS)))

    gset = g["u"].unique()
    rev = m_rows.filter(pl.col("u").is_in(gset))
    rev14 = rev.filter(~pl.col("tag").is_in(list(EXTRA_TAGS)))

    out = {
        "gold_unique_units": int(g["u"].n_unique()),
        "mix17_unique_units": int(m_uni["u"].n_unique()),
        "gold_to_mix17": {
            "colliding_gold_units": n_hit_units,
            "colliding_gold_claims": n_hit_owners,
            "fraction_of_gold_claims": round(n_hit_owners / n_gold_claims, 6),
            "per_tag": per_tag,
        },
        "gold_to_mix14": {
            "colliding_gold_units": int(h14["u"].n_unique()),
            "colliding_gold_claims": int(h14["owner_ix"].n_unique()),
            "fraction_of_gold_claims": round(h14["owner_ix"].n_unique() / n_gold_claims, 6),
        },
        "mix17_to_gold": {
            "colliding_mix_rows": int(rev.height),
            "colliding_mix_units": int(rev["u"].n_unique()),
            "fraction_of_mix_rows": round(rev.height / mix_rows_total, 6),
        },
        "mix14_to_gold": {
            "colliding_mix_rows": int(rev14.height),
            "colliding_mix_units": int(rev14["u"].n_unique()),
            "fraction_of_mix_rows": round(rev14.height / max(mix14_rows_total, 1), 6),
        },
    }
    log(f"  {label}: gold->mix17 {n_hit_units} units / {n_hit_owners} claims; "
        f"mix17->gold {rev.height} rows")
    return out


def _units(df, owner_col, cols, fn):
    """DataFrame -> (owner_ix, u) with u = fn applied and, for pairs, joined."""
    vals = [df[c].to_list() for c in cols]
    if len(cols) == 1:
        u = [fn(v) for v in vals[0]]
    else:
        u = [fn(a) + "\x00" + fn(b) for a, b in zip(*vals, strict=True)]
    d = {"u": u}
    if owner_col is not None:
        d["owner_ix"] = df[owner_col].to_list()
    return pl.DataFrame(d)


# --- document channel: 8-gram containment --------------------------------------


class DocIndex:
    """Sorted 8-gram hash index over the gold documents, with owning doc ids."""

    def __init__(self, docs, n):
        self.n = n
        self.hasher = G._TokenHasher()
        grams, owners, sizes = [], [], []
        for i, d in enumerate(docs):
            g = G.ngram_hashes(d, n, self.hasher)
            grams.append(g)
            owners.append(np.full(g.size, i, dtype=np.int64))
            sizes.append(g.size)
        self.gram = np.concatenate(grams)
        own = np.concatenate(owners)
        order = np.argsort(self.gram, kind="stable")
        self.gram = self.gram[order]
        self.owner = own[order]
        self.sizes = np.array(sizes, dtype=np.int64)
        self.hit = np.zeros(len(docs), dtype=np.int64)   # marked by scan()
        self.n_docs = len(docs)

    def scan(self, texts, mark=True):
        """Per-text containment INTO the doc universe; marks doc-side coverage."""
        cont = np.zeros(len(texts))
        scorable = np.zeros(len(texts), dtype=bool)
        seen = np.zeros(self.gram.size, dtype=bool)
        for i, t in enumerate(texts):
            q = G.ngram_hashes(t, self.n, self.hasher)
            if q.size == 0:
                continue
            scorable[i] = True
            lo = np.searchsorted(self.gram, q, side="left")
            hi = np.searchsorted(self.gram, q, side="right")
            nz = np.nonzero(hi > lo)[0]
            cont[i] = nz.size / q.size
            if mark and nz.size:
                for k in nz:
                    seen[lo[k]:hi[k]] = True
        if mark and seen.any():
            np.add.at(self.hit, self.owner[seen], 1)
        return cont, scorable

    def doc_coverage(self):
        return self.hit / np.maximum(self.sizes, 1)


def cont_summary(c, scorable):
    c = c[scorable] if scorable.any() else np.zeros(0)
    if c.size == 0:
        return {"n_scorable": 0}
    return {
        "n_scorable": int(c.size),
        "max": round(float(c.max()), 6),
        "p999": round(float(np.percentile(c, 99.9)), 6),
        "p99": round(float(np.percentile(c, 99)), 6),
        "mean": round(float(c.mean()), 6),
        "units_ge_0.30": int((c >= 0.30).sum()),
        "units_ge_0.50": int((c >= 0.50).sum()),
        "units_eq_1.00": int((c >= 0.999999).sum()),
    }


# --- C6 --------------------------------------------------------------------


def tok(s):
    return set(G.normalize(s).split())


def c6_feature(gold_claims, gold_keys, assoc):
    """max token-Jaccard between the gold claim and the claims the mix associates
    with that pair's key. Returns (feature, covered_mask)."""
    feat = np.zeros(len(gold_claims))
    cov = np.zeros(len(gold_claims), dtype=bool)
    cache = {}
    for i, (c, k) in enumerate(zip(gold_claims, gold_keys, strict=True)):
        a = assoc.get(k)
        if not a:
            continue
        cov[i] = True
        tc = cache.get(c)
        if tc is None:
            tc = cache[c] = tok(c)
        best = 0.0
        for other in a:
            to = cache.get(other)
            if to is None:
                to = cache[other] = tok(other)
            u = len(tc | to)
            if u:
                best = max(best, len(tc & to) / u)
        feat[i] = best
    return feat, cov


def main():
    res = {
        "surface": "gold_full",
        "class": "evaluation surface - in-domain held-out hold (bar >= 0.84)",
        "contract": "docs/experiments/dataset-contract.md (C1-C8), amendments C-A1 and C-A2 applied",
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
        "privacy": ("PRIVATE surface. No verbatim claim, evidence or document text and no "
                    "client or company name appears in this artifact. Counts, rates and "
                    "blake2b-64/128 fingerprints only."),
        "cpu_only": True,
        "hf_hub_offline": True,
    }

    # ---------------- assembly -------------------------------------------------
    gold, goldp, src, n_claims, y = assemble_gold()
    mix17 = assemble_mix17()
    mix14 = mix17.filter(~pl.col("tag").is_in(list(EXTRA_TAGS)))
    log(f"mix14 subset: {mix14.height} rows (expected {H150.EXPECTED_MIX_ROWS})")
    if mix14.height != H150.EXPECTED_MIX_ROWS:
        raise SystemExit("ABORT: 14-group subset is not the audit's mix")

    align = owner_alignment(gold, goldp, src)
    docs = src["source_text"].to_list()
    doc_uni = sorted(set(docs))

    res["assembly"] = {
        "gold_loader": "R10-H108_lane.gold_full()",
        "gold_source": str(GOLD_PAIRS.relative_to(ROOT)),
        "gold_document_source": str(GOLD_SRC.relative_to(ROOT)),
        "mix_loader": ("R10-H108_lane.public_train() under "
                       "R16-H142_G1_arm.untruncated_evidence(), plus lane parquets"),
        "mix17_rows": int(mix17.height),
        "mix17_lanes": "R20-H174_arm_run.LANES (5 lanes, 17 DANN groups)",
        "mix14_rows": int(mix14.height),
        "mix14_lanes": "R18-H150_arm_run.LANES (2 lanes, 14 DANN groups) - a strict tag-subset",
        "mix_ambiguity_note": (
            "The task brief names 721,210 rows AND R20-H174_arm_run.LANES; those are two "
            "different mixes (H150 lanes -> 721,210 / 14 groups, H174 lanes -> 760,618 / 17 "
            "groups). Both are measured here from one assembly and both are reported. The "
            "prior audit used the 14-group mix; the phase-1 member reports use the 17-group mix."),
        "owner_alignment": align,
    }
    res["surface_census"] = {
        "claims": n_claims,
        "pairs": int(gold.height),
        "unique_claim_strings": int(gold["claim"].n_unique()),
        "duplicate_claim_strings": n_claims - int(gold["claim"].n_unique()),
        "unique_chunks": int(gold["chunk"].n_unique()),
        "unique_documents": len(doc_uni),
        "trace_ids": int(src["trace_id"].n_unique()),
        "user_ids": int(src["user_id"].n_unique()),
        "label_base_rate": round(float(np.mean(y)), 4),
        "label_counts": {str(r["label"]): int(r["count"]) for r in src["label"].value_counts().sort("label").to_dicts()},
        "lang_counts": {r["lang"]: int(r["count"]) for r in src["lang"].value_counts().sort("lang").to_dicts()},
        "chunks_per_claim": {
            "min": int(gold.group_by("owner_ix").len()["len"].min()),
            "max": int(gold.group_by("owner_ix").len()["len"].max()),
            "mean": round(float(gold.group_by("owner_ix").len()["len"].mean()), 2),
        },
        "claims_per_document": {
            "min": int(src.group_by("source_text").len()["len"].min()),
            "max": int(src.group_by("source_text").len()["len"].max()),
            "mean": round(float(src.group_by("source_text").len()["len"].mean()), 2),
        },
        "document_chars": {
            "min": int(src["source_text"].str.len_chars().min()),
            "max": int(src["source_text"].str.len_chars().max()),
            "mean": int(src["source_text"].str.len_chars().mean()),
        },
        "fingerprints_blake2b_128": {
            "claims": fingerprint(gold["claim"].to_list()),
            "chunks": fingerprint(gold["chunk"].to_list()),
            "documents": fingerprint(doc_uni),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    log("=== STAGE ASSEMBLE DONE ===")

    # ---------------- STAGE 1: reproduce the banked audit ----------------------
    banked = json.loads(BANKED_AUDIT.read_text())
    mixwin14 = AUD.mix_windows(mix14)
    ex = AUD.exact_channels(gold, mix14, mixwin14, n_claims)
    prov = AUD.provenance(n_claims)
    idscan = AUD.id_substring_scan(mix14, prov)
    prov.pop("gold_ids")

    arena_claims14 = {t[0]: sorted({c for c in g["claim"].to_list() if c and c.strip()})
                      for t, g in mix14.group_by("tag")}
    gold_claim_units = sorted({c for c in gold["claim"].to_list() if c and c.strip()})
    cen_claims = AUD.census("repro_goldfull_claims", gold_claim_units, arena_claims14)

    arena_ev14 = {t[0]: sorted({w for w in g["window"].to_list() if w and w.strip()})
                  for t, g in mixwin14.group_by("tag")}
    gold_ev_units = sorted({c for c in gold["chunk"].to_list() if c and c.strip()})
    cen_ev = AUD.census("repro_goldfull_evidence", gold_ev_units, arena_ev14)

    # the audit's own live positive control: VitaminC TEST claims, same instrument
    z = zipfile.ZipFile(VITC)
    vt = pl.read_parquet(io.BytesIO(z.read(next(n for n in z.namelist() if n.endswith("__test.parquet")))))
    vt_claims = sorted({c for c in vt["claim"].to_list() if c and c.strip()})
    log(f"live positive control: {len(vt_claims)} VitaminC TEST claims vs the same 14-group mix")
    t = time.time()
    vt_res = G.run_gate(vt_claims, n=AUD.GATE_N, jaccard=AUD.GATE_JACCARD, kill=AUD.GATE_KILL,
                        label="vitaminc_test_claims", arena_texts=arena_claims14)
    log(f"live positive control: verdict {vt_res['verdict']} max_fraction {vt_res['max_fraction']} "
        f"units_with_hit {vt_res['candidate_vs_arena']['units_with_hit']}/{len(vt_claims)} "
        f"best_jaccard {vt_res['candidate_vs_arena'].get('best_jaccard')} ({time.time() - t:.1f}s)")

    def cmp(name, got, want):
        ok = got == want
        log(f"  repro {name}: got {got} / banked {want} -> {'MATCH' if ok else 'MISMATCH'}")
        return {"recomputed": got, "banked": want, "reproduces": bool(ok)}

    b = banked
    repro = {
        "banked_artifact": str(BANKED_AUDIT.relative_to(ROOT)),
        "banked_log": "logs/R20_goldfull_split_audit.log",
        "mix_used": "14-group / 721,210 rows - the audit's own mix",
        "checks": {
            "gold_claims": cmp("gold_claims", n_claims, b["sides"]["gold_full"]["claims"]),
            "gold_pairs": cmp("gold_pairs", int(gold.height), b["sides"]["gold_full"]["pairs"]),
            "gold_unique_claim_strings": cmp("gold_unique_claims", int(gold["claim"].n_unique()),
                                             b["sides"]["gold_full"]["unique_claim_strings"]),
            "gold_unique_chunks": cmp("gold_unique_chunks", int(gold["chunk"].n_unique()),
                                      b["sides"]["gold_full"]["unique_chunks"]),
            "gold_label_base_rate": cmp("gold_base_rate", round(float(np.mean(y)), 4),
                                        b["sides"]["gold_full"]["label_base_rate"]),
            "mix_rows": cmp("mix_rows", int(mix14.height), b["sides"]["training_mix"]["rows"]),
            "mix_unique_claims": cmp("mix_unique_claims", int(mix14["claim"].n_unique()),
                                     b["sides"]["training_mix"]["unique_claim_strings"]),
            "mix_unique_chunks": cmp("mix_unique_chunks", int(mix14["chunk"].n_unique()),
                                     b["sides"]["training_mix"]["unique_chunks"]),
            "mix_windows": cmp("mix_windows", int(mixwin14.height), b["sides"]["training_mix"]["windows"]),
            "mix_unique_windows": cmp("mix_unique_windows", int(mixwin14["window"].n_unique()),
                                      b["sides"]["training_mix"]["unique_windows"]),
            "exact_claims_colliding": cmp("exact_claims", ex["claims"]["colliding_gold_claims"],
                                          b["exact_match"]["claims"]["colliding_gold_claims"]),
            "exact_evidence_raw_colliding": cmp("exact_evidence_raw",
                                                ex["evidence_vs_raw_chunks"]["colliding_unique_chunks"],
                                                b["exact_match"]["evidence_vs_raw_chunks"]["colliding_unique_chunks"]),
            "exact_evidence_windows_colliding": cmp("exact_evidence_windows",
                                                    ex["evidence_vs_windows"]["colliding_unique_chunks"],
                                                    b["exact_match"]["evidence_vs_windows"]["colliding_unique_chunks"]),
            "exact_pairs_colliding": cmp("exact_pairs", ex["pairs"]["colliding_gold_pairs"],
                                         b["exact_match"]["pairs"]["colliding_gold_pairs"]),
            "id_substring_claim": cmp("id_substring_claim",
                                      idscan["mix_rows_containing_a_gold_id"]["claim"],
                                      b["provenance"]["id_substring_scan"]["mix_rows_containing_a_gold_id"]["claim"]),
            "id_substring_chunk": cmp("id_substring_chunk",
                                      idscan["mix_rows_containing_a_gold_id"]["chunk"],
                                      b["provenance"]["id_substring_scan"]["mix_rows_containing_a_gold_id"]["chunk"]),
            "id_join_possible": cmp("id_join_possible", prov["id_join_possible"],
                                    b["provenance"]["id_join_possible"]),
            "id_value_collisions": cmp("id_value_collisions", len(prov["id_value_collisions"]),
                                       b["summary"]["provenance_id_value_collisions"]),
            "census_claims_max_fraction": cmp("census_claims_max_fraction",
                                              cen_claims["result"]["max_fraction"],
                                              b["summary"]["near_duplicate_claims_max_fraction"]),
            "census_claims_best_jaccard_max": cmp("census_claims_best_jaccard",
                                                  cen_claims["result"]["candidate_vs_arena"]["best_jaccard"]["max"],
                                                  b["near_duplicate"]["claims"]["result"]["candidate_vs_arena"]["best_jaccard"]["max"]),
            "census_claims_verdict": cmp("census_claims_verdict", cen_claims["result"]["verdict"],
                                         b["summary"]["near_duplicate_claims_verdict"]),
            "census_evidence_max_fraction": cmp("census_evidence_max_fraction",
                                                cen_ev["result"]["max_fraction"],
                                                b["summary"]["near_duplicate_evidence_max_fraction"]),
            "census_evidence_best_jaccard_max": cmp("census_evidence_best_jaccard",
                                                    cen_ev["result"]["candidate_vs_arena"]["best_jaccard"]["max"],
                                                    b["near_duplicate"]["evidence"]["result"]["candidate_vs_arena"]["best_jaccard"]["max"]),
            "census_evidence_verdict": cmp("census_evidence_verdict", cen_ev["result"]["verdict"],
                                           b["summary"]["near_duplicate_evidence_verdict"]),
            "spike_claims_detected": cmp("spike_claims_detected", cen_claims["spike_control"]["detected_total"],
                                         b["near_duplicate"]["claims"]["spike_control"]["detected_total"]),
            "spike_evidence_detected": cmp("spike_evidence_detected", cen_ev["spike_control"]["detected_total"],
                                           b["near_duplicate"]["evidence"]["spike_control"]["detected_total"]),
            "live_control_max_fraction": cmp("live_control_max_fraction", vt_res["max_fraction"],
                                             b["summary"]["real_overlap_control_max_fraction"]),
            "live_control_units_with_hit": cmp("live_control_units_with_hit",
                                               vt_res["candidate_vs_arena"]["units_with_hit"],
                                               b["positive_controls"]["real_overlap_control"]["result"]["candidate_vs_arena"]["units_with_hit"]),
            "live_control_candidate_units": cmp("live_control_candidate_units", len(vt_claims),
                                                b["positive_controls"]["real_overlap_control"]["candidate_units"]),
            "live_control_best_jaccard_max": cmp("live_control_best_jaccard",
                                                 vt_res["candidate_vs_arena"]["best_jaccard"]["max"],
                                                 b["positive_controls"]["real_overlap_control"]["result"]["candidate_vs_arena"]["best_jaccard"]["max"]),
        },
    }
    repro["all_reproduce"] = all(v["reproduces"] for v in repro["checks"].values())
    repro["mismatches"] = [k for k, v in repro["checks"].items() if not v["reproduces"]]
    log(f"REPRODUCTION: all_reproduce={repro['all_reproduce']} mismatches={repro['mismatches']}")
    res["reproduction_of_prior_audit"] = repro
    res["C4_census"] = {
        "instrument": (f"provenance_gate.run_gate, R14-H136 ruling-2 form: {AUD.GATE_N}-gram, "
                       f"Jaccard >= {AUD.GATE_JACCARD}, bidirectional, KILL at {AUD.GATE_KILL:.0%}"),
        "claims_vs_mix14": {k: cen_claims["result"][k] for k in
                            ("max_fraction", "verdict", "candidate_vs_arena", "arena_vs_candidate")},
        "evidence_vs_mix14": {k: cen_ev["result"][k] for k in
                              ("max_fraction", "verdict", "candidate_vs_arena", "arena_vs_candidate")},
        "spike_controls": {"claims": cen_claims["spike_control"], "evidence": cen_ev["spike_control"]},
        "live_positive_control": {
            "candidate": "VitaminC official TEST split claims (public corpus), same instrument, same mix",
            "candidate_units": len(vt_claims),
            "units_with_hit": vt_res["candidate_vs_arena"]["units_with_hit"],
            "max_fraction": vt_res["max_fraction"],
            "best_jaccard": vt_res["candidate_vs_arena"]["best_jaccard"],
            "fires": vt_res["candidate_vs_arena"]["units_with_hit"] > 0,
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    del arena_claims14, arena_ev14
    log("=== STAGE 1 REPRODUCTION DONE ===")

    # ---------------- STAGE 2: C2 form matrix ----------------------------------
    mixwin17 = AUD.mix_windows(mix17)
    gold_doc_owner = pl.DataFrame({"owner_ix": src["owner_ix"].cast(pl.Int64).to_list(),
                                   "doc": docs})
    c2 = {"forms": FORM_DEFS,
          "directions": "gold->mix (fraction of the surface inside the mix) and mix->gold (fraction of mix rows inside the surface)",
          "channels": {}}
    channels = {
        "claims": (goldp.select(["owner_ix", "claim"]), ["claim"], mix17.select(["claim", "tag"]), ["claim"]),
        "evidence_vs_mix_chunks": (goldp.select(["owner_ix", "chunk"]), ["chunk"], mix17.select(["chunk", "tag"]), ["chunk"]),
        "evidence_vs_mix_windows": (goldp.select(["owner_ix", "chunk"]), ["chunk"],
                                    mixwin17.select(["window", "tag"]), ["window"]),
        "pairs": (goldp.select(["owner_ix", "claim", "chunk"]), ["claim", "chunk"],
                  mix17.select(["claim", "chunk", "tag"]), ["claim", "chunk"]),
        "documents_vs_mix_chunks": (gold_doc_owner, ["doc"], mix17.select(["chunk", "tag"]), ["chunk"]),
        "documents_vs_mix_windows": (gold_doc_owner, ["doc"], mixwin17.select(["window", "tag"]), ["window"]),
    }
    for cname, (gdf, gcols, mdf, mcols) in channels.items():
        c2["channels"][cname] = {}
        for fname, fn in FORMS.items():
            gu = _units(gdf, "owner_ix", gcols, fn)
            mu = _units(mdf, None, mcols, fn).with_columns(pl.Series("tag", mdf["tag"].to_list()))
            c2["channels"][cname][fname] = collide(f"{cname}/{fname}", gu, mu, n_claims)
        log(f"C2 channel {cname} done")
    OUT.write_text(json.dumps(res | {"C2": c2}, indent=2))

    # C2 live positive control: whitespace-remangled gold units injected into a
    # mix copy. Raw form must read 0; the ws-collapsed form must fire.
    rng = random.Random(20260817)
    k = 200
    inj_claims = rng.sample(sorted(set(gold["claim"].to_list())), k)
    mangled = [" \n ".join(s.split()) + "  " for s in inj_claims]
    poisoned = pl.DataFrame({"claim": mix17["claim"].to_list() + mangled,
                             "tag": mix17["tag"].to_list() + ["POISON"] * k})
    ctrl = {}
    for fname, fn in FORMS.items():
        gu = _units(goldp.select(["owner_ix", "claim"]), "owner_ix", ["claim"], fn)
        mu = _units(poisoned, None, ["claim"], fn).with_columns(pl.Series("tag", poisoned["tag"].to_list()))
        r = collide(f"C2-control/{fname}", gu, mu, n_claims)
        ctrl[fname] = {"colliding_gold_units": r["gold_to_mix17"]["colliding_gold_units"],
                       "per_tag": r["gold_to_mix17"]["per_tag"]}
    c2["live_positive_control"] = {
        "construction": (f"{k} gold claim strings re-wrapped (whitespace runs replaced by ' \\n ' "
                         "plus trailing spaces) and appended to a copy of the mix under tag POISON - "
                         "the R20-H177 failure mode by construction"),
        "injected_units": k,
        "per_form": ctrl,
        "raw_form_blind": ctrl["raw"]["colliding_gold_units"] == 0,
        "ws_form_fires": ctrl["ws_collapsed_casefold"]["colliding_gold_units"] == k,
    }
    log(f"C2 live positive control: raw {ctrl['raw']['colliding_gold_units']}, "
        f"truncated {ctrl['truncated_1500']['colliding_gold_units']}, "
        f"ws {ctrl['ws_collapsed_casefold']['colliding_gold_units']} of {k}")
    res["C2"] = c2
    OUT.write_text(json.dumps(res, indent=2))
    del poisoned
    log("=== STAGE 2 C2 DONE ===")

    # ---------------- STAGE 3: DOCUMENT channel (8-gram containment) -----------
    log(f"document channel: indexing {len(doc_uni)} gold documents at n={AUD.GATE_N}")
    di = DocIndex(doc_uni, AUD.GATE_N)
    log(f"document channel: {di.gram.size} document 8-grams indexed")

    mix_chunks_uni = sorted(set(mix17["chunk"].to_list()))
    log(f"document channel: scanning {len(mix_chunks_uni)} unique mix chunks")
    cont_mix, sc_mix = di.scan(mix_chunks_uni, mark=True)
    doc_cov = di.doc_coverage()
    log(f"document channel: mix->gold max containment {cont_mix.max():.6f}; "
        f"gold doc coverage max {doc_cov.max():.6f}")

    # true-parent control: gold chunks must be inside their own documents
    di2 = DocIndex(doc_uni, AUD.GATE_N)
    gold_chunks_uni = sorted(set(gold["chunk"].to_list()))
    cont_gold, sc_gold = di2.scan(gold_chunks_uni, mark=False)
    log(f"document channel true-parent control: gold chunk -> gold doc containment "
        f"mean {cont_gold[sc_gold].mean():.4f} max {cont_gold.max():.4f}")

    # manufactured spike: re-wrapped gold documents presented as mix chunks
    spike_docs = [" \n ".join(d.split()) for d in rng.sample(doc_uni, 10)]
    cont_spike, sc_spike = di2.scan(spike_docs, mark=False)
    log(f"document channel spike control: {int((cont_spike >= 0.999999).sum())}/10 at containment 1.0")

    res["document_channel"] = {
        "instrument": (f"{AUD.GATE_N}-gram containment (provenance_gate normalization and hashing); "
                       "Jaccard is not used because a 58k-char document against a 1.5k-char chunk "
                       "cannot reach any Jaccard threshold even at full containment"),
        "gold_documents": len(doc_uni),
        "gold_document_ngrams": int(di.gram.size),
        "mix_side": "unique untruncated mix chunks (a superset of the 1,500/750 windows: "
                    "every window n-gram is a chunk n-gram except at token-split boundaries, "
                    "so this direction upper-bounds the window presentation)",
        "mix_unique_chunks_scanned": len(mix_chunks_uni),
        "mix_chunk_into_gold_document": cont_summary(cont_mix, sc_mix),
        "gold_document_into_mix": {
            "n_docs": int(di.n_docs),
            "max": round(float(doc_cov.max()), 6),
            "p99": round(float(np.percentile(doc_cov, 99)), 6),
            "mean": round(float(doc_cov.mean()), 6),
            "docs_ge_0.30": int((doc_cov >= 0.30).sum()),
            "docs_ge_0.05": int((doc_cov >= 0.05).sum()),
        },
        "live_positive_control_true_parent": {
            "why": "the same instrument fed the genuine parent relation - gold chunks against "
                   "the gold documents they were cut from",
            "units": len(gold_chunks_uni),
            **cont_summary(cont_gold, sc_gold),
        },
        "spike_control_rewrapped_documents": {
            "injected": len(spike_docs),
            "detected_at_containment_1.0": int((cont_spike >= 0.999999).sum()),
            "passes": bool((cont_spike >= 0.999999).sum() == len(spike_docs)),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    del di, di2, mix_chunks_uni
    log("=== STAGE 3 DOCUMENT CHANNEL DONE ===")

    # ---------------- STAGE 4: C3 split axis -----------------------------------
    lane_sources = {}
    for fname, group, *_ in H174.LANES:
        d = pl.read_parquet(SEM / fname)
        lane_sources[group] = sorted(d["source"].unique().to_list()) if "source" in d.columns else []
    c2ch = c2["channels"]
    axis = {
        "question": "what axis is gold_full actually split on RELATIVE TO the training mix",
        "candidate_axes_measured": {
            "document": {
                "gold_documents": len(doc_uni),
                "documents_colliding_with_mix_any_form": max(
                    c2ch["documents_vs_mix_chunks"][f]["gold_to_mix17"]["colliding_gold_units"]
                    for f in FORMS) or 0,
                "max_ngram_containment_mix_chunk_into_gold_document":
                    res["document_channel"]["mix_chunk_into_gold_document"].get("max"),
                "max_ngram_coverage_gold_document_by_mix":
                    res["document_channel"]["gold_document_into_mix"]["max"],
            },
            "claim": {
                "colliding_gold_claims_any_form": max(
                    c2ch["claims"][f]["gold_to_mix17"]["colliding_gold_claims"] for f in FORMS),
                "near_duplicate_max_jaccard_vs_mix":
                    cen_claims["result"]["candidate_vs_arena"]["best_jaccard"]["max"],
            },
            "evidence": {
                "colliding_gold_chunks_any_form": max(
                    c2ch["evidence_vs_mix_chunks"][f]["gold_to_mix17"]["colliding_gold_units"] for f in FORMS),
                "near_duplicate_max_jaccard_vs_mix":
                    cen_ev["result"]["candidate_vs_arena"]["best_jaccard"]["max"],
            },
            "identifier": {
                "id_join_possible": prov["id_join_possible"],
                "id_value_collisions": len(prov["id_value_collisions"]),
                "gold_id_substring_hits_in_mix": idscan["mix_rows_containing_a_gold_id"],
            },
        },
        "mix_member_provenance": {
            "source_corpora": ["ragtruth_en", "ragtruth translated x7", "halueval",
                               "psiloqa", "vitaminc", "tabfact"],
            "constructed_lane_sources": lane_sources,
            "reading": ("every mix member and every constructed lane draws from a public corpus "
                        "(tabfact, feverous, vitaminc, minicheck, or a template generator); none "
                        "draws from the private corpus gold_full is cut from"),
        },
        "gold_internal_structure": {
            "claims": n_claims,
            "documents": len(doc_uni),
            "traces": int(src["trace_id"].n_unique()),
            "users": int(src["user_id"].n_unique()),
            "languages": int(src["lang"].n_unique()),
            "claims_share_a_document": int(n_claims - len(doc_uni)),
            "note": ("gold_full is consumed WHOLE by the in-domain suite, so it has no internal "
                     "train/test cut of its own; the axis question is entirely about its "
                     "relation to the mix"),
        },
    }
    OUT.write_text(json.dumps(res | {"C3_split_axis": axis}, indent=2))
    res["C3_split_axis"] = axis
    log("=== STAGE 4 C3 DONE ===")

    # ---------------- STAGE 5: C6 ----------------------------------------------
    log("C6: measuring key coverage for every field shared between gold_full and the mix")
    gold_claims_l = goldp["claim"].to_list()
    gold_chunks_l = goldp["chunk"].to_list()
    gold_owner_l = goldp["owner_ix"].to_list()
    owner_doc = dict(zip(src["owner_ix"].cast(pl.Int64).to_list(), docs, strict=True))
    gold_docs_l = [owner_doc[o] for o in gold_owner_l]
    pair_labels = goldp["label"].to_numpy()

    mix_claim_l = mix17["claim"].to_list()
    mix_chunk_l = mix17["chunk"].to_list()

    keydefs = {
        "evidence_raw": (gold_chunks_l, mix_chunk_l, f_raw),
        "evidence_ws": (gold_chunks_l, mix_chunk_l, f_ws),
        "document_raw": (gold_docs_l, mix_chunk_l, f_raw),
        "document_ws": (gold_docs_l, mix_chunk_l, f_ws),
        "claim_raw": (gold_claims_l, mix_claim_l, f_raw),
        "claim_ws": (gold_claims_l, mix_claim_l, f_ws),
    }
    c6 = {
        "clause_test": ("C6 is EVAL-FACING: for each pair, measure overlap between the eval claim "
                        "and whatever the TRAINING MIX associates with that pair's key (C-A2). "
                        "Computed here for every field gold_full and the mix could share"),
        "key_coverage": {},
    }
    assoc_cache = {}
    for kname, (gkeys, mkeys, fn) in keydefs.items():
        gk = [fn(s) for s in gkeys]
        mk_set = set()
        assoc = {}
        for mkey, mclaim in zip(mkeys, mix_claim_l, strict=True):
            u = fn(mkey)
            mk_set.add(u)
            assoc.setdefault(u, set()).add(mclaim)
        covered = sum(1 for u in gk if u in mk_set)
        c6["key_coverage"][kname] = {
            "gold_pairs": len(gk),
            "gold_pairs_with_key_in_mix": covered,
            "coverage": round(covered / len(gk), 6),
            "gold_distinct_keys": len(set(gk)),
            "mix_distinct_keys": len(mk_set),
            "distinct_keys_shared": len(set(gk) & mk_set),
        }
        log(f"C6 key {kname}: coverage {covered}/{len(gk)} pairs, "
            f"{len(set(gk) & mk_set)} distinct keys shared")
        assoc_cache[kname] = (gk, assoc)

    c6["identifier_keys"] = {
        "available": False,
        "reason": ("gold_full carries {owner (row index), trace_id, user_id, lang} and no "
                   "document / page / revision / corpus identifier; the mix members carry their "
                   "own namespaces. Measured absence, no proxy substituted (C-A2)."),
        "id_value_collisions": len(prov["id_value_collisions"]),
    }

    best_key = max(c6["key_coverage"], key=lambda k: c6["key_coverage"][k]["coverage"])
    if c6["key_coverage"][best_key]["coverage"] > 0:
        gk, assoc = assoc_cache[best_key]
        assoc = {k: sorted(v)[:64] for k, v in assoc.items()}
        feat, cov = c6_feature(gold_claims_l, gk, assoc)
        auroc = (round(float(roc_auc_score(pair_labels[cov], feat[cov])), 4)
                 if cov.sum() > 0 and len(set(pair_labels[cov].tolist())) == 2 else None)
        c6["measured_feature"] = {"key": best_key, "covered_pairs": int(cov.sum()),
                                  "auroc_vs_gold_label": auroc,
                                  "mean_feature": round(float(feat[cov].mean()), 4) if cov.sum() else None}
    else:
        c6["measured_feature"] = {
            "key": None,
            "status": "NOT-APPLICABLE - zero key coverage on every shared field",
            "reading": ("C6's eval-facing feature is UNDEFINED: no gold pair's key exists in the "
                        "training mix, so the mix associates nothing with any gold pair. Per C-A2 "
                        "no proxy is substituted."),
        }

    # C6 LIVE POSITIVE CONTROL - a mix-supplied association that is known-bad by
    # construction: the mix is made to associate the gold claim itself with the
    # positive legs' evidence key, and an unrelated gold claim with the negatives'.
    n_ctrl = 2000
    cand = rng.sample(range(len(gold_claims_l)), n_ctrl * 3)
    idx, seen_keys = [], set()
    for i in cand:
        k_ = gold_chunks_l[i]
        if k_ in seen_keys:
            continue
        seen_keys.add(k_)
        idx.append(i)
        if len(idx) == n_ctrl:
            break
    shuffled = [gold_claims_l[i] for i in idx]
    rng.shuffle(shuffled)
    poison_assoc = {}
    for j, i in enumerate(idx):
        poison_assoc[gold_chunks_l[i]] = [gold_claims_l[i]] if pair_labels[i] == 1 else [shuffled[j]]
    pf, pc = c6_feature([gold_claims_l[i] for i in idx], [gold_chunks_l[i] for i in idx], poison_assoc)
    pl_lab = pair_labels[idx]
    p_auroc = round(float(roc_auc_score(pl_lab[pc], pf[pc])), 4) if pc.sum() else None
    c6["live_positive_control"] = {
        "construction": (f"{n_ctrl} gold pairs sampled; a synthetic mix-supplied association is "
                         "built keyed on the pair's evidence, mapping the POSITIVE legs to their "
                         "own claim and the NEGATIVE legs to an unrelated gold claim - the "
                         "attr_pool failure mode by construction. In memory only."),
        "covered_pairs": int(pc.sum()),
        "coverage": round(float(pc.mean()), 4),
        "auroc_vs_gold_label": p_auroc,
        "fires": bool(p_auroc is not None and p_auroc >= 0.9),
    }
    log(f"C6 live positive control: coverage {pc.mean():.4f}, AUROC {p_auroc}")

    # separately-reported within-surface diagnostic (C-A2: a diagnostic, not a C6 bar)
    diag = {}
    for dname, keys in (("evidence", gold_chunks_l), ("document", gold_docs_l),
                        ("claim", gold_claims_l)):
        by = {}
        for k_, lab in zip(keys, pair_labels, strict=True):
            by.setdefault(k_, []).append(int(lab))
        base = float(pair_labels.mean())
        correct = 0
        n_pure = 0
        fallback = 1 if base >= 0.5 else 0
        for k_, labs in by.items():
            a = np.asarray(labs)
            n = a.size
            if n == 1:
                correct += int(fallback == a[0])
            else:
                loo = (a.sum() - a) / (n - 1)          # leave-one-out group mean
                correct += int(((loo >= 0.5).astype(int) == a).sum())
            if len(set(labs)) == 1:
                n_pure += n
        diag[dname] = {
            "distinct_keys": len(by),
            "rows": int(len(pair_labels)),
            "base_rate": round(base, 4),
            "majority_baseline": round(max(base, 1 - base), 4),
            "leave_one_out_key_lookup_accuracy": round(correct / len(pair_labels), 4),
            "rows_in_label_pure_groups": n_pure,
            "fraction_rows_label_pure": round(n_pure / len(pair_labels), 4),
        }
        log(f"C6 within-surface diagnostic ({dname}): LOO accuracy "
            f"{diag[dname]['leave_one_out_key_lookup_accuracy']} vs majority "
            f"{diag[dname]['majority_baseline']}")
    c6["within_surface_diagnostic"] = {
        "status": ("EXECUTOR-REPORTED DIAGNOSTIC, not a C6 bar (C-A2: a within-member "
                   "leave-one-out key lookup is a separately-reported diagnostic). It answers "
                   "whether rows inside gold_full predict each other, not whether the training "
                   "mix supplies the association."),
        "channels": diag,
    }
    res["C6"] = c6
    OUT.write_text(json.dumps(res, indent=2))
    log("=== STAGE 5 C6 DONE ===")

    # ---------------- summary (measurements, no adjudication) ------------------
    worst_c2_units = max(
        c2["channels"][ch][f]["gold_to_mix17"]["colliding_gold_units"]
        for ch in c2["channels"] for f in FORMS)
    worst_c2_rev = max(
        c2["channels"][ch][f]["mix17_to_gold"]["colliding_mix_rows"]
        for ch in c2["channels"] for f in FORMS)
    res["measurements"] = {
        "reproduction_all_match": repro["all_reproduce"],
        "reproduction_mismatches": repro["mismatches"],
        "C2_worst_colliding_gold_units_any_channel_any_form": int(worst_c2_units),
        "C2_worst_colliding_mix_rows_any_channel_any_form": int(worst_c2_rev),
        "C2_bar": "exactly zero on every form, both directions",
        "C2_clears": bool(worst_c2_units == 0 and worst_c2_rev == 0),
        "C4_claims_max_fraction": cen_claims["result"]["max_fraction"],
        "C4_evidence_max_fraction": cen_ev["result"]["max_fraction"],
        "C4_bar": f"KILL at {AUD.GATE_KILL}, WARN at 0.005",
        "C4_clears": bool(max(cen_claims["result"]["max_fraction"],
                              cen_ev["result"]["max_fraction"]) < 0.005),
        "document_channel_max_mix_chunk_containment":
            res["document_channel"]["mix_chunk_into_gold_document"].get("max"),
        "document_channel_max_gold_document_coverage":
            res["document_channel"]["gold_document_into_mix"]["max"],
        "C6_max_key_coverage": max(v["coverage"] for v in c6["key_coverage"].values()),
        "C6_bar": "a mix-supplied key association must not separate the classes; "
                  "zero coverage makes the clause's feature undefined (C-A2)",
        "positive_controls": {
            "C4_spike_claims": cen_claims["spike_control"]["passes"],
            "C4_spike_evidence": cen_ev["spike_control"]["passes"],
            "C4_live_vitaminc_test": res["C4_census"]["live_positive_control"]["fires"],
            "C2_ws_form_fires_on_rewrapped_gold": c2["live_positive_control"]["ws_form_fires"],
            "document_channel_true_parent": res["document_channel"]["live_positive_control_true_parent"].get("mean"),
            "document_channel_spike": res["document_channel"]["spike_control_rewrapped_documents"]["passes"],
            "C6_poisoned_association_auroc": c6["live_positive_control"]["auroc_vs_gold_label"],
        },
    }
    res["artifacts"] = [
        "experiments/grounding-semantic/contract/gold_full_surface.py",
        "experiments/grounding-semantic/contract/gold_full_surface_report.json",
        "logs/gold_full_surface.log",
    ]
    OUT.write_text(json.dumps(res, indent=2))
    log(f"summary: {json.dumps(res['measurements'])}")
    log(f"banked -> {OUT}")
    log("=== GOLD_FULL SURFACE VERIFICATION DONE ===")


if __name__ == "__main__":
    main()
