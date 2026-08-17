"""BLIND ARENA - dataset-contract verification, SURFACE SIDE.  CPU ONLY.

The blind arena is the campaign's frozen verdict surface: 10 RAGBench subsets,
2,264 responses, the R8-H77 frozen gate.  Every phase-1 member report checked
itself against the arena and read zero; this script does the check once from the
SURFACE side - the whole assembled mix against the whole arena, in every channel
the arena exposes.

WHAT IS MEASURED (nothing is adjudicated - the coordinator adjudicates):

  C2  disjointness.  Three arena channels (documents, responses, response
      sentences) against three mix channels (claims, raw evidence chunks, and
      the 1,500/750 windows the model was actually shown), in the campaign's
      eight string-form pairings crossed BOTH directions - a strict superset of
      the contract's "three string forms, both directions".  Per subset and per
      DANN group.  Then the DOCUMENT channel: arena document identifiers and
      their normalised stems against every provenance identifier field the mix
      members actually carry.
  C4  contamination census, the banked R14-H136 instrument (8-gram, Jaccard
      >= 0.3, bidirectional, KILL > 2%, per-corpus attribution), mix against
      arena, with THREE positive controls of which two are live near-duplicate
      text by construction, not synthetic spikes.
  SUB-DOCUMENT LEAKAGE.  Exact matching misses re-chunked text.  Per arena
      subset, the maximum 8-gram overlap FRACTION between any arena document
      and any mix chunk (max over single chunks, and the union over the whole
      mix), with the distribution's upper tail per subset.  Its own positive
      and negative controls.
  C3  split semantics.  Which split of each RAGBench subset the arena reads,
      read off the archive; a static scan of every loader in the mix-assembly
      chain for any RAGBench reference; and the empirical converse - every
      document of every OTHER RAGBench split of the same ten subsets checked
      against the assembled mix.

RAGBENCH CONTAMINATION WALL: RAGBench source corpora and derivatives are
FORBIDDEN as training data.  This script verifies the wall holds.

No evaluation-surface text is emitted - counts, identifiers and metrics only.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
          experiments/grounding-semantic/contract/arena_surface_verify.py \
          2>&1 | tee logs/contract-arena_surface.log
"""

import os

# CPU ONLY - GPU0/GPU1/GPU2 carry live training draws.  Set (not setdefault)
# BEFORE any banked module imports torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import collections
import importlib.util as _ilu
import io
import json
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).parent            # .../grounding-semantic/contract
EXP = HERE.parent                       # .../grounding-semantic
ROOT = EXP.parent.parent                # repo root
DATA = ROOT / "data" / "external" / "datasets"
ARCHIVE = DATA / "dataset-ragbench.zip"
OUT = HERE / "arena_surface_report.json"

NOTE = "Numbers recorded, not adjudicated - the coordinator adjudicates."

# Arena construction constants - byte-identical to R8-H77_unseen_arena.load_subsets()
MAX_CHUNKS = 8
N_PER_SUBSET = 250
EXPECT_SUBSETS = 10
EXPECT_RESPONSES = 2264

CLEAN_ROWS = 685_670
FLAGSHIP_ROWS = 721_210
PORTFOLIO_ROWS = 760_618
FLAGSHIP_ONLY_LANES = ("quant_misbind", "quant_scale_unit")
PORTFOLIO_EXTRA_LANES = ("frame_reject", "attr_pool", "path_bind")

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    m = _ilu.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading banked modules (CPU pinned)")
H143 = _mod("h143", EXP / "R17-H143_evalset_assessment.py")   # forms, cross_forms only
PG = _mod("provgate", EXP / "provenance_gate.py")             # the R14-H136 census instrument
ARM = _mod("g1arm", EXP / "R16-H142_G1_arm.py")               # windows(), untruncated_evidence, H108
H174 = _mod("h174", EXP / "R20-H174_arm_run.py")              # LANES, EXPECTED_GROUPS

_gates_src = (EXP / "R19_supply_gates.py").read_text()
GATE_N = int(_gates_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gates_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gates_src.split("GATE_KILL = ")[1].split("\n")[0])

norm = H143.norm


# --------------------------------------------------------------------------- #
# incremental banking
# --------------------------------------------------------------------------- #
def bank(res, key, value):
    res[key] = value
    OUT.write_text(json.dumps(res, indent=2, sort_keys=False) + "\n")
    log(f"banked -> {OUT.name} :: {key}")


# --------------------------------------------------------------------------- #
# the arena
# --------------------------------------------------------------------------- #
def load_arena():
    """The R8-H77 frozen gate's own construction, filter for filter, seed for seed.

    R8-H77_unseen_arena is NOT imported - it pins CUDA at import time.  The
    filter/sample chain below is copied from it verbatim and the resulting
    subset count and response count are asserted against the frozen figures.
    """
    z = zipfile.ZipFile(ARCHIVE)
    subsets = {}
    for name in sorted(n for n in z.namelist() if n.endswith("__test.parquet")):
        sub = name.split("__")[2]
        df = pl.read_parquet(io.BytesIO(z.read(name)))
        df = df.filter(
            pl.col("adherence_score").is_not_null()
            & (pl.col("response").str.len_chars() > 20)
            & (pl.col("documents").list.len() > 0)
        )
        if len(df) < 40 or df["adherence_score"].n_unique() < 2:
            log(f"arena: subset {sub} DROPPED by the frozen gate ({len(df)} rows)")
            continue
        df = df.sample(min(N_PER_SUBSET, len(df)), seed=0)
        docs = [d[:MAX_CHUNKS] for d in df["documents"].to_list()]
        subsets[sub] = {
            "file": name,
            "split": name.split("__")[3].replace(".parquet", ""),
            "ids": df["id"].cast(pl.Utf8).to_list(),
            "responses": df["response"].to_list(),
            "documents": docs,
            "sentences": [[s[1] for s in rs] for rs in df["response_sentences"].to_list()],
            "labels": df["adherence_score"].cast(pl.Int8).to_list(),
        }
    n_resp = sum(len(v["responses"]) for v in subsets.values())
    if len(subsets) != EXPECT_SUBSETS or n_resp != EXPECT_RESPONSES:
        raise SystemExit(
            f"ARENA ABORT: {len(subsets)} subsets / {n_resp} responses "
            f"!= frozen {EXPECT_SUBSETS} / {EXPECT_RESPONSES}")
    log(f"arena: {len(subsets)} subsets, {n_resp} responses (frozen gate reproduced)")
    return subsets


def arena_units(subsets):
    """Flat (unit, subset) lists for each of the three arena channels."""
    ch = {"documents": [], "responses": [], "response_sentences": []}
    owner = {k: [] for k in ch}
    for sub, v in subsets.items():
        for d in v["documents"]:
            for c in d:
                ch["documents"].append(c)
                owner["documents"].append(sub)
        for r in v["responses"]:
            ch["responses"].append(r)
            owner["responses"].append(sub)
        for ss in v["sentences"]:
            for s in ss:
                ch["response_sentences"].append(s)
                owner["response_sentences"].append(sub)
    return ch, owner


# --------------------------------------------------------------------------- #
# the mix
# --------------------------------------------------------------------------- #
def assemble():
    """The PORTFOLIO mix (a strict superset of the flagship), banked loaders only.

    NOT `R17-H143_evalset_assessment.build_mix`.  That helper loads
    `R16-H142_G1_arm` and `R10-H108_lane` as two SEPARATE module instances, so
    its `arm.untruncated_evidence()` patches `arm.H108.M59.CFG` while the
    `public_train()` it calls reads a different `M59.CFG` - the 1,500-char cut is
    never lifted and the clean mix comes back TRUNCATED (measured: 685,670
    windows instead of the banked 1,033,365).  Proven directly:
    `arm.M59 is arm.H108.M59` -> True, `arm.M59 is <separately loaded>.M59` ->
    False, and inside the context manager the two read 1,000,000,000 and 1,500.
    Recorded as an observation about that script; this run assembles the mix the
    way `R18-H150_arm_run.make_build_mix` does, through ONE module instance, and
    asserts the banked window geometry before measuring anything.
    """
    with ARM.untruncated_evidence():
        claims, chunks, y, tags = ARM.H108.public_train()
    claims, chunks, tags = list(claims), list(chunks), list(tags)
    labels = list(np.asarray(y, dtype="float64"))
    if len(claims) != CLEAN_ROWS:
        raise SystemExit(f"MIX ABORT: clean mix {len(claims)} rows != {CLEAN_ROWS}")
    log(f"mix: clean public {len(claims)} rows over {len(set(tags))} groups, UNTRUNCATED")

    for fname, group, n_rows, n_pairs, fams in H174.LANES:
        d = pl.read_parquet(EXP / fname)
        got_fams = {r["neg_family"]: int(r["count"])
                    for r in d["neg_family"].value_counts().to_dicts()}
        got_pairs = d["pair_id"].n_unique()
        if len(d) != n_rows or got_pairs != n_pairs or got_fams != fams:
            raise SystemExit(
                f"LANE ABORT ({group}): {len(d)} rows / {got_pairs} pairs {got_fams}")
        claims += d["claim"].to_list()
        chunks += d["chunk"].to_list()          # UNTRUNCATED - the twin protocol
        labels += [float(v) for v in d["label"].to_list()]
        tags += [group] * len(d)
        log(f"mix: lane {group} {len(d)} rows / {got_pairs} pairs {got_fams}")

    groups = tuple(sorted(set(tags)))
    if groups != H174.EXPECTED_GROUPS:
        raise SystemExit(f"GROUP-MAP ABORT: {groups} != {H174.EXPECTED_GROUPS}")
    if len(claims) != PORTFOLIO_ROWS:
        raise SystemExit(f"MIX ABORT: portfolio mix {len(claims)} rows != {PORTFOLIO_ROWS}")
    n_flag = sum(1 for t in tags if t not in PORTFOLIO_EXTRA_LANES)
    if n_flag != FLAGSHIP_ROWS:
        raise SystemExit(f"MIX ABORT: flagship sub-mix {n_flag} rows != {FLAGSHIP_ROWS}")
    cut = ARM.M59.CFG.chunk_max_chars
    log(f"mix: portfolio {len(claims)} rows / {len(groups)} groups; flagship sub-mix "
        f"{n_flag} rows; chunk_max_chars restored to {cut}")
    return claims, chunks, labels, tags, cut, list(groups)


# the banked window geometry this assembly must reproduce exactly
BANKED_CENSUS = {
    "clean_rows": 685_670, "clean_windows": 1_033_365,
    "portfolio_rows": 760_618, "portfolio_windows": 1_215_222,
    "portfolio_mean": 1.5977, "portfolio_multi_share": 0.2094,
    "flagship_mean": 1.4821, "flagship_multi_rows": 137_622,
}


def mix_windows(chunks, tags):
    win, wtag, counts = [], [], []
    for k, t in zip(chunks, tags, strict=True):
        ws = ARM.windows(k)
        counts.append(len(ws))
        win.extend(ws)
        wtag.extend([t] * len(ws))
    counts = np.array(counts)
    is_flag = np.array([t not in PORTFOLIO_EXTRA_LANES for t in tags])
    is_clean = np.array([t not in (PORTFOLIO_EXTRA_LANES + FLAGSHIP_ONLY_LANES)
                         for t in tags])
    geom = {
        "clean_rows": int(is_clean.sum()),
        "clean_windows": int(counts[is_clean].sum()),
        "portfolio_rows": len(counts),
        "portfolio_windows": int(counts.sum()),
        "portfolio_mean": round(float(counts.mean()), 4),
        "portfolio_multi_share": round(float((counts > 1).mean()), 4),
        "flagship_mean": round(float(counts[is_flag].mean()), 4),
        "flagship_multi_rows": int((counts[is_flag] > 1).sum()),
    }
    bad = {k: (geom[k], v) for k, v in BANKED_CENSUS.items() if geom[k] != v}
    if bad:
        raise SystemExit(f"WINDOW CENSUS ABORT: measured != banked {bad}")
    log(f"mix windows: {len(win)} over {len(chunks)} rows (mean "
        f"{geom['portfolio_mean']}) - reproduces the banked R20-H174 census exactly")
    return win, wtag, geom


# --------------------------------------------------------------------------- #
# C2 - exact disjointness in eight string-form pairings, both directions
# --------------------------------------------------------------------------- #
def attribute(hit_units, mix_texts, mix_tags, cut):
    """For a set of colliding ARENA unit strings, name the mix groups that carry
    them.  Built only when a hit exists, so a clean run pays nothing."""
    if not hit_units:
        return {}
    targets = {}
    for form in ("raw", "trunc", "nraw", "ntrunc"):
        targets[form] = collections.defaultdict(set)
    for t, g in zip(mix_texts, mix_tags, strict=True):
        if not t:
            continue
        targets["raw"][t].add(g)
        targets["trunc"][t[:cut]].add(g)
        targets["nraw"][norm(t)].add(g)
        targets["ntrunc"][norm(t[:cut])].add(g)
    per_group = collections.Counter()
    for u in hit_units:
        forms = {"raw": u, "trunc": u[:cut], "nraw": norm(u), "ntrunc": norm(u[:cut])}
        gs = set()
        for f, val in forms.items():
            gs |= targets[f].get(val, set())
        for g in gs:
            per_group[g] += 1
    return dict(per_group)


def c2_exact(arena_ch, arena_owner, claims, chunks, windows, ctags, wtags, cut):
    mix_channels = {
        "mix_claims": (claims, ctags),
        "mix_evidence_raw_chunks": (chunks, ctags),
        "mix_evidence_windows": (windows, wtags),
    }
    forms = {name: H143.form_sets(texts, cut) for name, (texts, _) in mix_channels.items()}
    for name in forms:
        log(f"C2: mix form-sets built for {name} "
            f"({len(forms[name]['raw'])} distinct raw units)")

    out = {}
    for ach, units in arena_ch.items():
        out[ach] = {"arena_units": len(units), "arena_units_distinct": len(set(units))}
        for mch, fs in forms.items():
            counts, hit, _ = H143.cross_forms(units, fs, cut, reverse=True)
            total_fwd = len(hit)
            block = {
                "form_pairings": counts,
                "arena_units_hit_any_form": total_fwd,
                "arena_fraction": round(total_fwd / max(len(set(units)), 1), 8),
            }
            if hit:
                mtexts, mtags = mix_channels[mch]
                block["per_mix_group"] = attribute(hit, mtexts, mtags, cut)
                bysub = collections.Counter()
                for u, s in zip(arena_ch[ach], arena_owner[ach], strict=True):
                    if u in hit:
                        bysub[s] += 1
                block["per_arena_subset"] = dict(bysub)
            else:
                block["per_mix_group"] = {}
                block["per_arena_subset"] = {}
            out[ach][mch] = block
            log(f"C2: arena {ach:20s} vs {mch:24s} -> "
                f"{total_fwd} distinct arena units in the mix "
                f"(reverse max "
                f"{max(c.get('mix_units_in_eval', 0) for c in counts.values())})")

        # per-subset forward-only pass (reverse is a whole-mix scan; run once above)
        per_sub = {}
        for sub in sorted(set(arena_owner[ach])):
            u = [x for x, s in zip(arena_ch[ach], arena_owner[ach], strict=True) if s == sub]
            row = {"units": len(u), "units_distinct": len(set(u))}
            for mch, fs in forms.items():
                counts, hit, _ = H143.cross_forms(u, fs, cut, reverse=False)
                row[mch] = {"arena_units_hit_any_form": len(hit),
                            "by_form": {k: v["eval_units_in_mix"] for k, v in counts.items()}}
            per_sub[sub] = row
        out[ach]["per_subset"] = per_sub
        log(f"C2: per-subset pass done for {ach}")
    return out


# --------------------------------------------------------------------------- #
# C2 - document channel (identifiers, not text)
# --------------------------------------------------------------------------- #
_ALNUM = re.compile(r"[^a-z0-9]+")
_TITLE = re.compile(r"^\s*Title:\s*(.+?)\s*$", re.MULTILINE)
_URL = re.compile(r"https?://[^\s\"'<>]+")


def stem(s):
    """Normalised identifier stem: casefold, strip the TabFact `1-`/`2-` csv-id
    prefix and `.html.csv` suffix, then drop every non-alphanumeric character."""
    t = s.strip().casefold()
    if t[:2] in ("1-", "2-"):
        t = t[2:]
    for suf in (".html.csv", ".csv", ".html"):
        if t.endswith(suf):
            t = t[: -len(suf)]
    return _ALNUM.sub("", t)


def arena_doc_ids(subsets):
    """Every document-level identifier the arena actually exposes."""
    ids = {"row_id": set(), "subset_row_id": set(), "document_title": set(),
           "document_url": set()}
    for sub, v in subsets.items():
        for rid, docs in zip(v["ids"], v["documents"], strict=True):
            ids["row_id"].add(rid)
            ids["subset_row_id"].add(f"{sub}_{rid}")
            for d in docs:
                for m in _TITLE.findall(d):
                    ids["document_title"].add(m)
                first = d.split("\n", 1)[0]
                for u in _URL.findall(first):
                    ids["document_url"].add(u)
    return ids


def mix_provenance_ids():
    """Every provenance identifier field the mix members actually carry."""
    out = {}
    id_cols = {"id", "unique_id", "case_id", "wiki_revision_id", "page", "fever_id",
               "table_id", "wiki_title", "wiki_url", "doc_id", "row_key", "source",
               "swap_doc_id", "pool_doc_ids"}
    archives = {
        "ragtruth_en": ("dataset-ragtruth.zip", "__train.parquet"),
        "ragtruth_translated": ("dataset-ragtruth-translated.zip", "__train.parquet"),
        "psiloqa": ("dataset-psiloqa.zip", "__train.parquet"),
        "vitaminc": ("dataset-vitaminc.zip", "__train.parquet"),
        "tabfact": ("dataset-tabfact.zip", "__train.parquet"),
    }
    for member, (zn, suf) in archives.items():
        z = zipfile.ZipFile(DATA / zn)
        for h in [x for x in z.namelist() if x.endswith(suf)]:
            df = pl.read_parquet(io.BytesIO(z.read(h)))
            for c in df.columns:
                if c.lower() in id_cols:
                    out.setdefault(f"{member}.{c}", set()).update(
                        df[c].cast(pl.Utf8).drop_nulls().to_list())
    # halueval: public_train reads the qa and summarization configs, which carry
    # no identifier column at all - recorded explicitly rather than skipped
    zh = zipfile.ZipFile(DATA / "dataset-halueval.zip")
    for cfg in ("qa", "summarization"):
        hits = [x for x in zh.namelist() if f"__{cfg}__" in x]
        if hits:
            df = pl.read_parquet(io.BytesIO(zh.read(hits[0])))
            cols = [c for c in df.columns if c.lower() in id_cols]
            for c in cols:
                out.setdefault(f"halueval_{cfg}.{c}", set()).update(
                    df[c].cast(pl.Utf8).drop_nulls().to_list())
            if not cols:
                out.setdefault(f"halueval_{cfg}.<none>", set())
    for fname, group in H143.DOC_LANES.items():
        p = EXP / fname
        if not p.exists():
            continue
        d = pl.read_parquet(p)
        for c in d.columns:
            if c.lower() in id_cols:
                col = d[c]
                if col.dtype == pl.List(pl.String):
                    vals = [x for lst in col.to_list() if lst is not None for x in lst]
                else:
                    vals = col.cast(pl.Utf8).drop_nulls().to_list()
                out.setdefault(f"{group}.{c}", set()).update(vals)
    return out


def doc_channel(subsets):
    a_ids = arena_doc_ids(subsets)
    m_ids = mix_provenance_ids()
    log(f"doc channel: arena id families {[f'{k}={len(v)}' for k, v in a_ids.items()]}")
    log(f"doc channel: mix provenance fields {len(m_ids)}")

    res = {"arena_identifier_families": {k: len(v) for k, v in a_ids.items()},
           "mix_provenance_fields": {k: len(v) for k, v in sorted(m_ids.items())},
           "collisions": {}, "collisions_stem": {}}
    numericish = re.compile(r"^\d+$")
    for an, av in a_ids.items():
        av_stem = {stem(x) for x in av if stem(x)}
        for mn, mv in m_ids.items():
            if not mv:
                continue
            raw_hit = av & mv
            stem_hit = av_stem & {stem(x) for x in mv if stem(x)}
            if raw_hit:
                res["collisions"][f"{an} x {mn}"] = {
                    "n": len(raw_hit),
                    "all_bare_numeric_both_sides": bool(
                        all(numericish.match(x) for x in raw_hit)),
                    "sample": sorted(raw_hit)[:10],
                }
            if stem_hit:
                res["collisions_stem"][f"{an} x {mn}"] = {
                    "n": len(stem_hit),
                    "all_bare_numeric_both_sides": bool(
                        all(numericish.match(x) for x in stem_hit)),
                    "sample": sorted(stem_hit)[:10],
                }
    log(f"doc channel: {len(res['collisions'])} raw id-family collisions, "
        f"{len(res['collisions_stem'])} stem collisions")
    return res


# --------------------------------------------------------------------------- #
# n-gram instrument - one pass over the mix, all channels
# --------------------------------------------------------------------------- #
def _max_pair(query, sorted_hashes, owner, unit_sizes):
    """Best Jaccard AND best containment of `query` against any single indexed
    unit, from one search.  Jaccard is `provenance_gate._max_jaccard` verbatim;
    containment replaces the union denominator with the query's own size, which
    is what detects a re-chunked fragment that Jaccard's length penalty hides."""
    if query.size == 0 or sorted_hashes.size == 0:
        return 0.0, 0.0, -1, -1
    lo = np.searchsorted(sorted_hashes, query, side="left")
    hi = np.searchsorted(sorted_hashes, query, side="right")
    nz = np.nonzero(hi > lo)[0]
    if nz.size == 0:
        return 0.0, 0.0, -1, -1
    ids = np.concatenate([owner[lo[i]: hi[i]] for i in nz])
    uids, inter = np.unique(ids, return_counts=True)
    union = query.size + unit_sizes[uids] - inter
    j = inter / np.maximum(union, 1)
    c = inter / query.size
    kj, kc = int(np.argmax(j)), int(np.argmax(c))
    return float(j[kj]), float(c[kc]), int(uids[kj]), int(uids[kc])


def _index(hash_list):
    flat = np.concatenate(hash_list) if hash_list else np.empty(0, dtype=np.uint64)
    owner = (np.concatenate([np.full(a.size, i, dtype=np.int64)
                             for i, a in enumerate(hash_list)])
             if hash_list else np.empty(0, dtype=np.int64))
    order = np.argsort(flat, kind="stable")
    sizes = np.array([a.size for a in hash_list], dtype=np.int64)
    return flat[order], owner[order], sizes


def census(arena_ch, arena_owner, chunks, tags, n=GATE_N, thr=GATE_JACCARD):
    """The R14-H136 gate (8-gram, Jaccard >= thr, bidirectional, per-corpus
    attribution) plus the sub-document containment reading, in ONE pass over the
    mix.  Both directions; the reverse side is attributed to the arena subset."""
    hasher = PG._TokenHasher()

    # arena side
    q, q_owner = {}, {}
    for ch, units in arena_ch.items():
        uniq = sorted(set(units))
        sub_of = {}
        for u, s in zip(units, arena_owner[ch], strict=True):
            sub_of.setdefault(u, s)
        q[ch] = [PG.ngram_hashes(u, n, hasher) for u in uniq]
        q_owner[ch] = [sub_of[u] for u in uniq]
        scor = sum(1 for a in q[ch] if a.size)
        log(f"census: arena {ch}: {len(uniq)} distinct units, {scor} scorable at {n}-grams")

    # arena DOCUMENT index for the reverse direction (one bucket; owner -> subset)
    doc_flat, doc_owner, doc_sizes = _index(q["documents"])
    doc_sub = q_owner["documents"]

    by_group = collections.defaultdict(set)
    for c, t in zip(chunks, tags, strict=True):
        if c:
            by_group[t].add(c)
    log(f"census: mix has {sum(len(v) for v in by_group.values())} distinct chunks "
        f"over {len(by_group)} groups")

    best_j = {ch: np.zeros(len(q[ch])) for ch in q}
    best_c = {ch: np.zeros(len(q[ch])) for ch in q}
    best_j_grp = {ch: [None] * len(q[ch]) for ch in q}
    best_c_grp = {ch: [None] * len(q[ch]) for ch in q}
    union_c = {ch: np.zeros(len(q[ch])) for ch in q}       # union over the WHOLE mix
    union_c_grp = {ch: [None] * len(q[ch]) for ch in q}
    per_group = {}
    rev_hits = collections.Counter()
    rev_by_group = {}
    global_hashes = np.empty(0, dtype=np.uint64)

    for grp in sorted(by_group):
        t0 = time.time()
        units = sorted(by_group[grp])
        hs = [PG.ngram_hashes(u, n, hasher) for u in units]
        flat, owner, sizes = _index(hs)
        guniq = np.unique(flat)

        gstats = {"mix_units": len(units),
                  "mix_units_scorable": int(sum(1 for a in hs if a.size)),
                  "distinct_ngrams": int(guniq.size)}
        for ch in q:
            hits = 0
            gmaxj = 0.0
            for i, qq in enumerate(q[ch]):
                if qq.size == 0:
                    continue
                j, c, _uj, _uc = _max_pair(qq, flat, owner, sizes)
                if j > best_j[ch][i]:
                    best_j[ch][i], best_j_grp[ch][i] = j, grp
                if c > best_c[ch][i]:
                    best_c[ch][i], best_c_grp[ch][i] = c, grp
                gmaxj = max(gmaxj, j)
                hits += j >= thr
                # per-group UNION containment (this group's whole n-gram pool)
                lo = np.searchsorted(guniq, qq, side="left")
                hi = np.searchsorted(guniq, qq, side="right")
                u = float((hi > lo).sum()) / qq.size
                if u > union_c[ch][i]:
                    union_c[ch][i], union_c_grp[ch][i] = u, grp
            gstats[f"{ch}_units_at_jaccard_ge_thr"] = int(hits)
            gstats[f"{ch}_max_jaccard"] = round(gmaxj, 4)

        # reverse: every mix unit of this group against the arena DOCUMENT index
        rev = 0
        rev_subs = collections.Counter()
        for a in hs:
            if a.size == 0:
                continue
            j, _c, uj, _uc = _max_pair(a, doc_flat, doc_owner, doc_sizes)
            if j >= thr:
                rev += 1
                rev_subs[doc_sub[uj]] += 1
        gstats["mix_units_at_jaccard_ge_thr_vs_arena_documents"] = int(rev)
        gstats["reverse_hits_by_arena_subset"] = dict(rev_subs)
        rev_hits[grp] = rev
        rev_by_group[grp] = dict(rev_subs)

        global_hashes = np.union1d(global_hashes, guniq)
        per_group[grp] = gstats
        log(f"census: {grp:18s} units={len(units):7d} ngrams={guniq.size:10d} "
            f"docs_j>=thr={gstats['documents_units_at_jaccard_ge_thr']:4d} "
            f"max_j={gstats['documents_max_jaccard']:.4f} rev={rev:5d} "
            f"({time.time() - t0:.0f}s)")
        del hs, flat, owner, sizes, guniq

    log(f"census: whole-mix distinct {n}-grams = {global_hashes.size}")
    whole_union = {}
    for ch in q:
        v = np.zeros(len(q[ch]))
        for i, qq in enumerate(q[ch]):
            if qq.size == 0:
                continue
            lo = np.searchsorted(global_hashes, qq, side="left")
            hi = np.searchsorted(global_hashes, qq, side="right")
            v[i] = float((hi > lo).sum()) / qq.size
        whole_union[ch] = v
    cen = {
        "q_sizes": {ch: np.array([a.size for a in q[ch]], dtype=np.int64) for ch in q},
        "q_owner": q_owner, "best_j": best_j, "best_c": best_c,
        "best_j_grp": best_j_grp, "best_c_grp": best_c_grp,
        "union_c": union_c, "union_c_grp": union_c_grp,
        "whole_union": whole_union, "per_group": per_group,
        "rev_hits": dict(rev_hits), "rev_by_group": rev_by_group,
        "global_hashes": global_hashes, "hasher": hasher,
    }
    save_census(cen)
    return cen


CACHE = ROOT / "tmp" / "arena_surface"


def save_census(cen):
    """Checkpoint the census to disk - the pass over the mix costs ~12 minutes and
    must never be repeated because a later stage raised."""
    CACHE.mkdir(parents=True, exist_ok=True)
    arrays = {"global_hashes": cen["global_hashes"]}
    for key in ("q_sizes", "best_j", "best_c", "union_c", "whole_union"):
        for ch, v in cen[key].items():
            arrays[f"{key}|{ch}"] = np.asarray(v)
    np.savez(CACHE / "census.npz", **arrays)
    meta = {k: cen[k] for k in ("q_owner", "best_j_grp", "best_c_grp", "union_c_grp",
                                "per_group", "rev_hits", "rev_by_group")}
    (CACHE / "census_meta.json").write_text(json.dumps(meta))
    log(f"census checkpointed -> {CACHE}")


def load_census():
    p = CACHE / "census.npz"
    if not p.exists() or not (CACHE / "census_meta.json").exists():
        return None
    z = np.load(p)
    cen = json.loads((CACHE / "census_meta.json").read_text())
    for key in ("q_sizes", "best_j", "best_c", "union_c", "whole_union"):
        cen[key] = {}
    for k in z.files:
        if k == "global_hashes":
            cen["global_hashes"] = z[k]
            continue
        key, ch = k.split("|", 1)
        cen[key][ch] = z[k]
    cen["hasher"] = PG._TokenHasher()
    log(f"census restored from {CACHE} "
        f"({cen['global_hashes'].size} distinct mix 8-grams)")
    return cen


def c4_block(cen, arena_ch, thr=GATE_JACCARD):
    out = {
        "instrument": (f"provenance_gate primitives, {GATE_N}-gram, "
                       f"Jaccard >= {GATE_JACCARD}, bidirectional, per-corpus "
                       f"attribution (the banked R14-H136 ruling-2 form)"),
        "kill_threshold": GATE_KILL,
        "per_mix_group": cen["per_group"],
    }
    for ch in cen["q_sizes"]:
        n_units = int(len(cen["q_sizes"][ch]))
        scor = int((cen["q_sizes"][ch] > 0).sum())
        bj = cen["best_j"][ch]
        hit = int((bj >= thr).sum())
        per_sub = collections.Counter()
        per_sub_n = collections.Counter()
        for s, v in zip(cen["q_owner"][ch], bj, strict=True):
            per_sub_n[s] += 1
            if v >= thr:
                per_sub[s] += 1
        out[ch] = {
            "arena_units_distinct": n_units,
            "arena_units_scorable": scor,
            "arena_units_too_short_for_ngram": n_units - scor,
            "arena_units_at_jaccard_ge_thr": hit,
            "fraction_of_distinct_units": round(hit / max(n_units, 1), 8),
            "best_jaccard": {"max": round(float(bj.max()), 4),
                             "p999": round(float(np.percentile(bj, 99.9)), 4),
                             "p99": round(float(np.percentile(bj, 99)), 4),
                             "mean": round(float(bj.mean()), 6)},
            "per_arena_subset_units": dict(per_sub_n),
            "per_arena_subset_hits": dict(per_sub),
            "per_arena_subset_hit_fraction": {
                s: round(per_sub.get(s, 0) / per_sub_n[s], 6) for s in per_sub_n},
        }
    rev_total = int(sum(cen["rev_hits"].values()))
    out["reverse_direction"] = {
        "mix_units_at_jaccard_ge_thr_vs_arena_documents": rev_total,
        "per_mix_group": cen["rev_hits"],
        "per_mix_group_by_arena_subset": cen["rev_by_group"],
    }
    fwd_frac = max(out[ch]["fraction_of_distinct_units"] for ch in cen["q_sizes"])
    mix_units = sum(v["mix_units"] for v in cen["per_group"].values())
    rev_frac = rev_total / max(mix_units, 1)
    out["max_fraction"] = round(max(fwd_frac, rev_frac), 8)
    out["reverse_fraction"] = round(rev_frac, 8)
    out["verdict_vs_kill_bar"] = "KILL" if out["max_fraction"] >= GATE_KILL else (
        "WARN" if out["max_fraction"] >= 0.005 else "PASS")
    return out


def subdoc_block(cen, arena_ch, arena_owner):
    """Sub-document leakage: 8-gram overlap FRACTION, per arena subset, with tail."""
    out = {}
    for ch in ("documents", "responses", "response_sentences"):
        cols = {"subset": [], "max_single_chunk": [], "max_single_group": [],
                "whole_mix_union": [], "best_chunk_group": [], "best_group": []}
        sub = cen["q_owner"][ch]
        for i in range(len(cen["q_sizes"][ch])):
            if cen["q_sizes"][ch][i] == 0:
                continue
            cols["subset"].append(sub[i])
            cols["max_single_chunk"].append(float(cen["best_c"][ch][i]))
            cols["max_single_group"].append(float(cen["union_c"][ch][i]))
            cols["whole_mix_union"].append(float(cen["whole_union"][ch][i]))
            cols["best_chunk_group"].append(cen["best_c_grp"][ch][i] or "")
            cols["best_group"].append(cen["union_c_grp"][ch][i] or "")
        df = pl.DataFrame(cols, schema={"subset": pl.Utf8, "max_single_chunk": pl.Float64,
                                        "max_single_group": pl.Float64,
                                        "whole_mix_union": pl.Float64,
                                        "best_chunk_group": pl.Utf8,
                                        "best_group": pl.Utf8})
        per = {}
        for s in sorted(df["subset"].unique().to_list()):
            d = df.filter(pl.col("subset") == s)
            def tail(col):
                a = d[col].to_numpy()
                return {
                    "max": round(float(a.max()), 4),
                    "p999": round(float(np.percentile(a, 99.9)), 4),
                    "p99": round(float(np.percentile(a, 99)), 4),
                    "p95": round(float(np.percentile(a, 95)), 4),
                    "median": round(float(np.median(a)), 4),
                    "mean": round(float(a.mean()), 4),
                    "n_ge_0.10": int((a >= 0.10).sum()),
                    "n_ge_0.25": int((a >= 0.25).sum()),
                    "n_ge_0.50": int((a >= 0.50).sum()),
                    "n_ge_0.90": int((a >= 0.90).sum()),
                }
            am = d["max_single_chunk"].to_numpy()
            k = int(np.argmax(am))
            per[s] = {
                "scorable_units": d.height,
                "max_overlap_fraction_vs_any_single_mix_chunk": tail("max_single_chunk"),
                "max_overlap_fraction_vs_any_single_mix_group_pool": tail("max_single_group"),
                "overlap_fraction_vs_whole_mix_union": tail("whole_mix_union"),
                "argmax_unit_attribution": {
                    "max_single_chunk": round(float(am[k]), 4),
                    "mix_group": d["best_chunk_group"].to_list()[k],
                },
                "group_of_max_single_chunk_counts": {
                    r["best_chunk_group"]: int(r["len"])
                    for r in d.group_by("best_chunk_group").len()
                    .sort("len", descending=True).iter_rows(named=True)
                    if r["best_chunk_group"]},
            }
        out[ch] = {"per_subset": per,
                   "note": ("`max_single_chunk` is |ngrams(arena unit) INTERSECT "
                            "ngrams(one mix chunk)| / |ngrams(arena unit)|; "
                            "`whole_mix_union` pools every mix chunk, so a document "
                            "re-chunked ACROSS several mix rows still registers")}
    return out


def exposure_block(cen, subsets, arena_ch, thr=GATE_JACCARD):
    """Which arena documents fire, and how much of the arena's SCORED SURFACE they
    touch.  The arena's scoring unit is the response, so a document-level hit
    matters in proportion to the responses that retrieve it."""
    import hashlib
    uniq = sorted(set(arena_ch["documents"]))
    bj = cen["best_j"]["documents"]
    hit_ix = [i for i in range(len(uniq)) if bj[i] >= thr]
    hit_docs = {uniq[i] for i in hit_ix}
    detail = []
    for i in sorted(hit_ix, key=lambda k: -bj[k]):
        d = uniq[i]
        detail.append({
            "arena_subset": cen["q_owner"]["documents"][i],
            "doc_blake2b_64": hashlib.blake2b(d.encode("utf-8"), digest_size=8).hexdigest(),
            "chars": len(d), "tokens": len(d.split()),
            "max_jaccard": round(float(bj[i]), 4),
            "jaccard_mix_group": cen["best_j_grp"]["documents"][i],
            "max_containment_single_chunk": round(float(cen["best_c"]["documents"][i]), 4),
            "containment_mix_group": cen["best_c_grp"]["documents"][i],
            "whole_mix_union_containment": round(
                float(cen["whole_union"]["documents"][i]), 4),
        })
    per_sub = {}
    for sub, v in subsets.items():
        n_resp = len(v["responses"])
        touched, lab_touched = 0, []
        for docs, lab in zip(v["documents"], v["labels"], strict=True):
            if any(c in hit_docs for c in docs):
                touched += 1
                lab_touched.append(int(lab))
        per_sub[sub] = {
            "responses": n_resp,
            "responses_retrieving_a_hit_document": touched,
            "fraction_of_subset_responses": round(touched / n_resp, 6),
            "fraction_of_whole_arena_responses": round(touched / EXPECT_RESPONSES, 6),
            "grounded_rate_all_responses": round(float(np.mean(v["labels"])), 4),
            "grounded_rate_touched_responses": (
                round(float(np.mean(lab_touched)), 4) if lab_touched else None),
        }
    tot = sum(v["responses_retrieving_a_hit_document"] for v in per_sub.values())
    log(f"exposure: {len(hit_ix)} hit documents touch {tot}/{EXPECT_RESPONSES} arena "
        f"responses ({tot / EXPECT_RESPONSES:.4f})")
    return {
        "definition": (f"a hit document is an arena document reading Jaccard >= {thr} "
                       f"against some single mix chunk under the banked "
                       f"{GATE_N}-gram instrument"),
        "hit_documents": len(hit_ix),
        "arena_responses_retrieving_a_hit_document": tot,
        "fraction_of_arena_responses": round(tot / EXPECT_RESPONSES, 6),
        "per_subset": per_sub,
        "per_document": detail,
        "load_bearing_reading": (
            "NOT MEASURED HERE. Deciding whether the banked arena AUROC moves when "
            "these responses are dropped needs per-row arena scores for the banked "
            "checkpoints; producing them is a GPU read and all three cards carry "
            "live training draws. The exposure counts above are what bounds it."),
    }


# --------------------------------------------------------------------------- #
# positive / negative controls
# --------------------------------------------------------------------------- #
def controls(cen, chunks, tags, arena_ch, n=GATE_N, thr=GATE_JACCARD, k=10):
    """Every gate gets a live positive control fed something known-bad.

    1  SPIKE (synthetic) - real mix chunks injected on the arena side.
    2  LIVE RE-CHUNK - real mix chunks re-wrapped at a different window offset,
       so no exact form matches but the text is near-duplicate by construction.
       This is the exact transformation that defeats exact matching, and it is
       the positive control for BOTH the Jaccard gate and the containment
       instrument.
    3  LIVE CROSS-SPLIT - VitaminC's own official test/validation evidence,
       genuine near-duplicate text by construction: VitaminC train supplies
       370,653 of the mix's rows and its official split shares pages, claims and
       evidence strings with train (the R20 baseline-leg finding).
    4  NEGATIVE - arena documents with their tokens randomly permuted, which
       destroys every 8-gram while preserving the vocabulary.
    """
    hasher = cen["hasher"]
    glob = cen["global_hashes"]

    # a fresh per-group index is expensive; the controls are scored against the
    # WHOLE-MIX n-gram pool for containment and against a 60k-chunk sample index
    # for Jaccard, which is the same instrument on a subset of the same corpus
    rng = np.random.default_rng(0)
    pool = [c for c in dict.fromkeys(chunks) if c]
    idx_sample = pool if len(pool) <= 60000 else [
        pool[i] for i in rng.choice(len(pool), 60000, replace=False)]
    hs = [PG.ngram_hashes(u, n, hasher) for u in idx_sample]
    flat, owner, sizes = _index(hs)
    log(f"controls: Jaccard index built over {len(idx_sample)} mix chunks")

    long_units = [u for u in idx_sample if len(u.split()) > 200][:200]
    if len(long_units) < k:
        long_units = [u for u in idx_sample if len(u.split()) > 60][:200]

    def score(units):
        js, cs, us = [], [], []
        for u in units:
            qq = PG.ngram_hashes(u, n, hasher)
            if qq.size == 0:
                continue
            j, c, _a, _b = _max_pair(qq, flat, owner, sizes)
            lo = np.searchsorted(glob, qq, side="left")
            hi = np.searchsorted(glob, qq, side="right")
            js.append(round(j, 4)); cs.append(round(c, 4))
            us.append(round(float((hi > lo).sum()) / qq.size, 4))
        return js, cs, us

    # 1 - spike
    sj, sc, su = score(long_units[:k])
    spike = {"kind": "synthetic spike - real mix chunks injected on the arena side",
             "n": len(sj), "jaccards": sj, "containments": sc,
             "detected_at_jaccard_ge_thr": int(sum(j >= thr for j in sj)),
             "fires": bool(sj) and all(j >= thr for j in sj)}
    log(f"controls: spike {spike['detected_at_jaccard_ge_thr']}/{spike['n']} fire")

    # 2 - live re-chunk (different window boundaries, no exact form match)
    rec = []
    for u in long_units[:k]:
        cut = min(len(u) - 1, 1500)
        off = 137 if len(u) > 1637 else max(1, len(u) // 7)
        frag = u[off: off + cut]
        if len(frag.split()) >= n:
            rec.append(frag)
    rj, rc, ru = score(rec)
    exact_forms = set(idx_sample) | {c[:1500] for c in idx_sample} | {norm(c) for c in idx_sample}
    rechunk = {
        "kind": ("LIVE near-duplicate by construction - real mix chunks re-wrapped "
                 "at a different window offset; no exact string form matches, the "
                 "text is the corpus's own"),
        "n": len(rj),
        "exact_form_matches": int(sum(
            1 for f in rec if f in exact_forms or f[:1500] in exact_forms
            or norm(f) in exact_forms)),
        "jaccards": rj, "containments": rc, "whole_mix_union_containments": ru,
        "detected_at_jaccard_ge_thr": int(sum(j >= thr for j in rj)),
        "fires": bool(rj) and all(j >= thr for j in rj),
        "containment_fires": bool(rc) and all(c >= 0.9 for c in rc),
    }
    log(f"controls: live re-chunk {rechunk['detected_at_jaccard_ge_thr']}/{rechunk['n']} "
        f"fire on Jaccard, containment min "
        f"{min(rc) if rc else None}, exact matches {rechunk['exact_form_matches']}")

    # 3 - live cross-split: VitaminC official test/validation evidence
    zv = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    vc = []
    for split in ("test", "validation"):
        nm = [x for x in zv.namelist() if x.endswith(f"__{split}.parquet")]
        if nm:
            d = pl.read_parquet(io.BytesIO(zv.read(nm[0])))
            vc += d["evidence"].drop_nulls().to_list()
    vc_u = sorted(set(vc))
    vj, vc_c, vu = score(vc_u)
    cross = {
        "kind": ("LIVE near-duplicate by construction - VitaminC's own official "
                 "test+validation evidence against the mix VitaminC train "
                 "supplies 370,653 rows of; the official split is disjoint by "
                 "unique_id/case_id but shares pages, claims and evidence text"),
        "n_units": len(vc_u), "n_scorable": len(vj),
        "detected_at_jaccard_ge_thr": int(sum(j >= thr for j in vj)),
        "fraction": round(sum(j >= thr for j in vj) / max(len(vj), 1), 6),
        "max_jaccard": round(max(vj), 4) if vj else None,
        "max_containment": round(max(vc_c), 4) if vc_c else None,
        "units_at_containment_ge_0.9": int(sum(c >= 0.9 for c in vc_c)),
        "fires": bool(vj) and any(j >= thr for j in vj),
    }
    log(f"controls: live cross-split VitaminC {cross['detected_at_jaccard_ge_thr']}"
        f"/{cross['n_scorable']} at Jaccard>=thr, max {cross['max_jaccard']}")

    # 4 - negative: arena documents with tokens permuted
    negs = []
    for u in arena_ch["documents"][:400]:
        t = u.split()
        if len(t) < 60:
            continue
        rng.shuffle(t)
        negs.append(" ".join(t))
        if len(negs) >= k:
            break
    nj, nc, nu = score(negs)
    negative = {
        "kind": ("NEGATIVE - arena documents with their tokens randomly permuted; "
                 "vocabulary preserved, every 8-gram destroyed"),
        "n": len(nj), "jaccards": nj, "containments": nc,
        "whole_mix_union_containments": nu,
        "detected_at_jaccard_ge_thr": int(sum(j >= thr for j in nj)),
        "silent": bool(nj) and not any(j >= thr for j in nj),
    }
    log(f"controls: negative {negative['detected_at_jaccard_ge_thr']}/{negative['n']} fire "
        f"(want 0), max containment {max(nc) if nc else None}")
    return {"spike_control": spike, "live_rechunk_control": rechunk,
            "live_cross_split_control": cross, "negative_control": negative}


# --------------------------------------------------------------------------- #
# C3 - split semantics
# --------------------------------------------------------------------------- #
def c3_block(subsets, chunks, claims, cut):
    z = zipfile.ZipFile(ARCHIVE)
    splits_present = collections.defaultdict(list)
    for nm in z.namelist():
        if nm.endswith(".parquet"):
            parts = nm.split("__")
            splits_present[parts[2]].append(parts[3].replace(".parquet", ""))

    loaders = ["R10-H108_lane.py", "R16-H142_G1_arm.py", "R18-H150_arm_run.py",
               "R20-H174_arm_run.py", "R7-H59_cross_domain_matrix.py"]
    scan = {}
    for f in loaders:
        p = EXP / f
        if not p.exists():
            scan[f] = "ABSENT"
            continue
        src = p.read_text()
        hits = [ln.strip() for ln in src.splitlines() if "ragbench" in ln.lower()]
        scan[f] = {"ragbench_mentions": len(hits),
                   "lines": [h[:200] for h in hits[:6]]}

    # empirical converse: every document of the OTHER RAGBench splits vs the mix
    mix_forms = {"raw": set(chunks), "trunc": {c[:cut] for c in chunks},
                 "nraw": {norm(c) for c in chunks}}
    mix_claim_forms = {"raw": set(claims), "nraw": {norm(c) for c in claims}}
    other = {}
    for sub in sorted(subsets):
        row = {}
        for split in ("train", "validation", "test"):
            nm = f"galileo-ai__ragbench__{sub}__{split}.parquet"
            if nm not in z.namelist():
                row[split] = None
                continue
            d = pl.read_parquet(io.BytesIO(z.read(nm)))
            docs = {c for lst in d["documents"].to_list() if lst is not None for c in lst}
            resp = {r for r in d["response"].to_list() if r}
            row[split] = {
                "rows": d.height, "distinct_documents": len(docs),
                "documents_in_mix_raw": len(docs & mix_forms["raw"]),
                "documents_in_mix_truncated": len({x[:cut] for x in docs} & mix_forms["trunc"]),
                "documents_in_mix_normalised": len({norm(x) for x in docs} & mix_forms["nraw"]),
                "responses_in_mix_claims_raw": len(resp & mix_claim_forms["raw"]),
                "responses_in_mix_claims_normalised": len(
                    {norm(x) for x in resp} & mix_claim_forms["nraw"]),
            }
        # within-corpus: does the arena's test split share documents with the others?
        arena_docs = {c for d in subsets[sub]["documents"] for c in d}
        for split in ("train", "validation"):
            nm = f"galileo-ai__ragbench__{sub}__{split}.parquet"
            if nm in z.namelist():
                d = pl.read_parquet(io.BytesIO(z.read(nm)))
                docs = {c for lst in d["documents"].to_list() if lst is not None for c in lst}
                row[f"arena_documents_shared_with_{split}"] = len(arena_docs & docs)
        other[sub] = row
        log(f"C3: {sub:12s} splits={sorted(splits_present[sub])} "
            f"train-docs-in-mix={other[sub]['train']['documents_in_mix_raw'] if other[sub].get('train') else 'n/a'}")

    return {
        "arena_reads": {s: subsets[s]["split"] for s in sorted(subsets)},
        "arena_source_files": {s: subsets[s]["file"] for s in sorted(subsets)},
        "splits_present_in_archive": {k: sorted(v) for k, v in sorted(splits_present.items())},
        "static_scan_of_mix_loaders": scan,
        "other_splits_vs_assembled_mix": other,
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-census", action="store_true")
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse the checkpointed n-gram census instead of redoing "
                         "the ~12-minute pass over the mix")
    args = ap.parse_args()

    res = {
        "artifact": "arena_surface_report.json",
        "surface": "the blind arena - 10 RAGBench subsets, 2,264 responses, "
                   "the R8-H77 frozen gate",
        "question": ("does any part of the assembled training mix touch the blind "
                     "arena, in any channel, in any string form, at any granularity"),
        "contamination_wall": ("RAGBench source corpora and derivatives are FORBIDDEN "
                               "as training data; this run verifies the wall, it does "
                               "not test it"),
        "note": NOTE,
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT.write_text(json.dumps(res, indent=2) + "\n")

    subsets = load_arena()
    arena_ch, arena_owner = arena_units(subsets)
    claims, chunks, labels, tags, cut, groups = assemble()
    windows, wtags, geom = mix_windows(chunks, tags)

    bank(res, "sides", {
        "arena": {
            "subsets": sorted(subsets),
            "responses": sum(len(v["responses"]) for v in subsets.values()),
            "per_subset_responses": {s: len(subsets[s]["responses"]) for s in sorted(subsets)},
            "documents_total": len(arena_ch["documents"]),
            "documents_distinct": len(set(arena_ch["documents"])),
            "responses_distinct": len(set(arena_ch["responses"])),
            "response_sentences_total": len(arena_ch["response_sentences"]),
            "response_sentences_distinct": len(set(arena_ch["response_sentences"])),
            "construction": ("R8-H77_unseen_arena.load_subsets() reproduced filter for "
                             f"filter: adherence_score non-null, response > 20 chars, "
                             f">= 1 document, subsets with < 40 rows or one label "
                             f"dropped, sample(min({N_PER_SUBSET}, n), seed=0), first "
                             f"{MAX_CHUNKS} documents per response"),
        },
        "mix": {
            "assembly": ("R10-H108_lane.public_train() read UNTRUNCATED through "
                         "R16-H142_G1_arm.untruncated_evidence() (one module "
                         "instance, the way R18-H150_arm_run.make_build_mix does "
                         "it) plus the five lanes of R20-H174_arm_run.LANES, "
                         "each verified on rows, pairs and negative-family counts"),
            "window_geometry_vs_banked_R20_H174_census": geom,
            "clean_public_rows": CLEAN_ROWS,
            "flagship_rows": FLAGSHIP_ROWS,
            "portfolio_rows": len(claims),
            "dann_groups": groups,
            "flagship_groups": [g for g in groups if g not in PORTFOLIO_EXTRA_LANES],
            "portfolio_only_groups": list(PORTFOLIO_EXTRA_LANES),
            "distinct_claims": len(set(claims)),
            "distinct_chunks": len(set(chunks)),
            "windows": len(windows),
            "distinct_windows": len(set(windows)),
            "chunk_max_chars": cut,
            "window": {"size": ARM.WIN, "stride": ARM.STRIDE},
        },
    })

    bank(res, "executor_observations", {
        "R17-H143_evalset_assessment.build_mix_truncates_the_clean_mix": {
            "what": ("that helper loads R16-H142_G1_arm and R10-H108_lane as two "
                     "separate module instances, so its `arm.untruncated_evidence()` "
                     "patches arm.H108.M59.CFG while the public_train() it calls "
                     "reads a different M59.CFG object"),
            "measured": ("arm.M59 is arm.H108.M59 -> True; arm.M59 is "
                         "<separately loaded H108>.M59 -> False; inside the context "
                         "manager the two chunk_max_chars read 1000000000 and 1500. "
                         "The mix it returns windows to 867,527 windows / mean "
                         "1.1406 against the banked 1,215,222 / 1.5977"),
            "consequence": ("that script's C2 evidence channel was measured against "
                            "clean-mix text cut at 1,500 chars, so any collision "
                            "living past that cut was outside its reach. This run "
                            "does NOT use it - the mix here reproduces the banked "
                            "R20-H174 window census exactly"),
            "scope": "an observation about a different artifact; nothing is adjudicated here",
        },
        "note": NOTE,
    })

    bank(res, "C2_exact_disjointness",
         c2_exact(arena_ch, arena_owner, claims, chunks, windows, tags, wtags, cut))
    bank(res, "C2_document_channel", doc_channel(subsets))
    bank(res, "C3_split_semantics", c3_block(subsets, chunks, claims, cut))

    if args.skip_census:
        log("census skipped by flag")
        return

    cen = load_census() if args.use_cache else None
    if cen is None:
        cen = census(arena_ch, arena_owner, chunks, tags)
    bank(res, "C4_contamination_census", c4_block(cen, arena_ch))
    bank(res, "C4_exposure", exposure_block(cen, subsets, arena_ch))
    bank(res, "sub_document_leakage", subdoc_block(cen, arena_ch, arena_owner))
    bank(res, "positive_controls", controls(cen, chunks, tags, arena_ch))

    # --- roll-up ---------------------------------------------------------- #
    c2 = res["C2_exact_disjointness"]
    worst_exact = max(
        c2[a][m]["arena_units_hit_any_form"]
        for a in c2 for m in ("mix_claims", "mix_evidence_raw_chunks",
                              "mix_evidence_windows"))
    worst_rev = max(
        v["mix_units_in_eval"]
        for a in c2 for m in ("mix_claims", "mix_evidence_raw_chunks",
                              "mix_evidence_windows")
        for v in c2[a][m]["form_pairings"].values())
    c4 = res["C4_contamination_census"]
    bank(res, "summary", {
        "C2_exact_max_arena_units_in_mix_any_channel_any_form": worst_exact,
        "C2_exact_max_mix_units_in_arena_any_channel_any_form": worst_rev,
        "C2_document_channel_raw_collisions": len(res["C2_document_channel"]["collisions"]),
        "C2_document_channel_stem_collisions": len(
            res["C2_document_channel"]["collisions_stem"]),
        "C4_max_fraction": c4["max_fraction"],
        "C4_kill_bar": GATE_KILL,
        "C4_verdict_vs_bar": c4["verdict_vs_kill_bar"],
        "C4_documents_max_jaccard": c4["documents"]["best_jaccard"]["max"],
        "C4_documents_per_subset_hit_fraction": c4["documents"][
            "per_arena_subset_hit_fraction"],
        "C4_reverse_mix_units_hit": c4["reverse_direction"][
            "mix_units_at_jaccard_ge_thr_vs_arena_documents"],
        "C4_arena_responses_retrieving_a_hit_document": res["C4_exposure"][
            "arena_responses_retrieving_a_hit_document"],
        "C4_fraction_of_arena_responses_exposed": res["C4_exposure"][
            "fraction_of_arena_responses"],
        "subdoc_max_single_chunk_overlap_by_subset": {
            s: v["max_overlap_fraction_vs_any_single_mix_chunk"]["max"]
            for s, v in res["sub_document_leakage"]["documents"]["per_subset"].items()},
        "subdoc_whole_mix_union_overlap_by_subset": {
            s: v["overlap_fraction_vs_whole_mix_union"]["max"]
            for s, v in res["sub_document_leakage"]["documents"]["per_subset"].items()},
        "controls_all_fire": bool(
            res["positive_controls"]["spike_control"]["fires"]
            and res["positive_controls"]["live_rechunk_control"]["fires"]
            and res["positive_controls"]["live_cross_split_control"]["fires"]
            and res["positive_controls"]["negative_control"]["silent"]),
        "note": NOTE,
    })
    log("=== ARENA SURFACE VERIFICATION COMPLETE ===")


if __name__ == "__main__":
    main()
