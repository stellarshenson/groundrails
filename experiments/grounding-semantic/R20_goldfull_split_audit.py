"""R20 - split audit: is `gold_full` disjoint from the assembled training mix?

WHY.  A baseline-leg build found that VitaminC's official train/test/validation
split is disjoint by `unique_id` / `case_id` but NOT by page, claim text,
evidence text or revision.  In this data family an "official split" does not
imply text disjointness.  `gold_full` - the in-domain hold every campaign arm
carries - has never been audited that way.  This script measures the overlap; it
adjudicates nothing.

SIDES.
  gold_full   `R10-H108_lane.gold_full()` - ALL 2,752 gold claims from
              `private-rag-forensics/R7-H51_teacher_pairs.parquet`, each claim
              carrying its chunk list.  Served unit at read time is
              `chunk[:CFG.chunk_max_chars]` (`R16-H142_G1_arm.score_claims`);
              every gold chunk is already <= 1,500 chars, so served == raw.
  training    `R10-H108_lane.public_train()` read UNTRUNCATED through
              `R16-H142_G1_arm.untruncated_evidence()`, plus the two lanes named
              in `R18-H150_arm_run.LANES`, presented as 1,500/750 windows
              (`R16-H142_G1_arm.windows`).  Banked loaders, not reimplemented.

CHANNELS.
  exact       claim strings, evidence strings, (claim, evidence) pairs; raw
              chunk text and, separately, the windows the model was actually
              shown.  Per-corpus attribution by DANN group tag.
  near-dup    `provenance_gate.run_gate` in the R14-H136 ruling-2 form - 8-gram,
              Jaccard >= 0.3, bidirectional, with the spike control - run with
              the TRAINING MIX standing in for the arena side, so its per-bucket
              breakdown attributes any hit to a corpus.  Instrument reused, not
              rewritten; thresholds read from `R19_supply_gates.py`.
  provenance  the identifier fields that actually exist on each side, and the
              collisions between them.

CPU ONLY - CUDA_VISIBLE_DEVICES is forced empty before any import, so no card is
touched.  Polars throughout.

Run:  uv run python experiments/grounding-semantic/R20_goldfull_split_audit.py \
          2>&1 | tee logs/R20_goldfull_split_audit.log
"""

import os

# Hard CPU pin - set (not setdefault) BEFORE the banked modules import torch.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import importlib.util
import io
import json
import pathlib
import time
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
CACHE = ROOT / "tmp" / "R20_goldfull_split_audit"
OUT = HERE / "R20_goldfull_split_audit.json"

GOLD_PAIRS = HERE / "private-rag-forensics" / "R7-H51_teacher_pairs.parquet"
GOLD_SRC = HERE / "private-rag-forensics" / "gold" / "golden_grounding_evidence_verified.parquet"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


log("loading banked modules (CPU, CUDA_VISIBLE_DEVICES='')")
ARM = _mod("g1arm", "R16-H142_G1_arm.py")      # untruncated_evidence, windows, H108
H150 = _mod("h150", "R18-H150_arm_run.py")     # LANES, EXPECTED_* census constants
G = _mod("provgate", "provenance_gate.py")     # the R14-H136 census instrument
H108 = ARM.H108
M59 = ARM.M59

_gates_src = (HERE / "R19_supply_gates.py").read_text()
GATE_N = int(_gates_src.split("GATE_N = ")[1].split("\n")[0])
GATE_JACCARD = float(_gates_src.split("GATE_JACCARD = ")[1].split("\n")[0])
GATE_KILL = float(_gates_src.split("GATE_KILL = ")[1].split("\n")[0])
SPIKE_SAMPLE = 2000

SERVE_CHARS = M59.CFG.chunk_max_chars


# --- assembly ------------------------------------------------------------------


def assemble_mix():
    """The H150/H174 mix: banked clean loader untruncated + the registered lanes."""
    with ARM.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    log(f"clean public mix: {len(y)} rows (expected {H150.EXPECTED_CLEAN_ROWS})")
    if len(y) != H150.EXPECTED_CLEAN_ROWS:
        raise SystemExit("CENSUS ABORT: clean mix row count is not the incumbent's")

    for fname, group, n_rows, n_pairs, fams in H150.LANES:
        df = pl.read_parquet(HERE / fname)
        got_fams = {r["neg_family"]: int(r["count"]) for r in df["neg_family"].value_counts().to_dicts()}
        got_pairs = df["pair_id"].n_unique()
        if len(df) != n_rows or got_pairs != n_pairs or got_fams != fams:
            raise SystemExit(f"LANE ABORT ({group}): {len(df)} rows / {got_pairs} pairs {got_fams}")
        claims += df["claim"].to_list()
        chunks += df["chunk"].to_list()
        y = np.concatenate([y, df["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * len(df)
        log(f"lane {group}: {len(df)} rows, {got_pairs} pairs, {got_fams}")

    names = tuple(sorted(set(tags)))
    if names != H150.EXPECTED_GROUPS:
        raise SystemExit(f"GROUP-MAP ABORT: {names}")
    log(f"assembled mix: {len(y)} rows (expected {H150.EXPECTED_MIX_ROWS}), {len(names)} groups")
    return pl.DataFrame({"claim": claims, "chunk": chunks, "tag": tags}), len(y)


def assemble_gold():
    """gold_full exactly as the in-domain suite reads it."""
    cl, ck, y = H108.gold_full()
    rows = {"owner_ix": [], "claim": [], "chunk": []}
    for i, (c, ks) in enumerate(zip(cl, ck, strict=True)):
        for k in ks:
            rows["owner_ix"].append(i)
            rows["claim"].append(c)
            rows["chunk"].append(k[:SERVE_CHARS])  # the served unit
    df = pl.DataFrame(rows)
    log(f"gold_full: {len(y)} claims, {df.height} (claim, chunk) rows, "
        f"{df['claim'].n_unique()} unique claims, {df['chunk'].n_unique()} unique chunks")
    return df, len(y), y


def mix_windows(mix):
    """The 1,500/750 presentation - the text the model was actually shown."""
    win, tag = [], []
    for k, t in zip(mix["chunk"].to_list(), mix["tag"].to_list(), strict=True):
        ws = ARM.windows(k)
        win.extend(ws)
        tag.extend([t] * len(ws))
    df = pl.DataFrame({"window": win, "tag": tag})
    log(f"mix windows: {df.height} windows over {mix.height} rows "
        f"(mean {df.height / mix.height:.4f}), {df['window'].n_unique()} unique")
    return df


# --- exact-match channels ------------------------------------------------------


def _attr(hit_df, col):
    """Per-corpus attribution: colliding gold units -> contributing DANN groups."""
    return {r[col]: int(r["len"]) for r in hit_df.group_by(col).len().sort("len", descending=True).iter_rows(named=True)}


def exact_channels(gold, mix, mixwin, n_claims):
    res = {}
    n_rows = gold.height
    n_uc = gold["claim"].n_unique()
    n_uk = gold["chunk"].n_unique()

    # --- claims
    mc = mix.select(["claim", "tag"]).unique()
    hit = gold.select(["owner_ix", "claim"]).unique().join(mc, on="claim", how="inner")
    res["claims"] = {
        "gold_unique_claim_strings": n_uc,
        "gold_claims": n_claims,
        "colliding_unique_claim_strings": int(hit["claim"].n_unique()),
        "colliding_gold_claims": int(hit["owner_ix"].n_unique()),
        "fraction_of_gold_claims": round(hit["owner_ix"].n_unique() / n_claims, 6),
        "per_corpus": _attr(hit, "tag"),
    }
    log(f"exact claims: {res['claims']['colliding_gold_claims']}/{n_claims} gold claims "
        f"({res['claims']['fraction_of_gold_claims']:.6f})")

    # --- evidence, raw mix chunk text
    mk = mix.select(["chunk", "tag"]).unique()
    hit = gold.join(mk, on="chunk", how="inner")
    res["evidence_vs_raw_chunks"] = {
        "gold_unique_chunks": n_uk,
        "gold_rows": n_rows,
        "colliding_unique_chunks": int(hit["chunk"].n_unique()),
        "colliding_gold_rows": int(hit.select(["owner_ix", "chunk"]).unique().height),
        "colliding_gold_claims": int(hit["owner_ix"].n_unique()),
        "fraction_of_gold_claims": round(hit["owner_ix"].n_unique() / n_claims, 6),
        "fraction_of_gold_unique_chunks": round(hit["chunk"].n_unique() / n_uk, 6),
        "per_corpus": _attr(hit, "tag"),
    }
    log(f"exact evidence (raw): {res['evidence_vs_raw_chunks']['colliding_unique_chunks']}/{n_uk} "
        f"unique gold chunks")

    # --- evidence, against the windows actually shown
    mw = mixwin.unique()
    hit = gold.join(mw, left_on="chunk", right_on="window", how="inner")
    res["evidence_vs_windows"] = {
        "mix_unique_windows": int(mw["window"].n_unique()),
        "colliding_unique_chunks": int(hit["chunk"].n_unique()),
        "colliding_gold_claims": int(hit["owner_ix"].n_unique()),
        "fraction_of_gold_claims": round(hit["owner_ix"].n_unique() / n_claims, 6),
        "fraction_of_gold_unique_chunks": round(hit["chunk"].n_unique() / n_uk, 6),
        "per_corpus": _attr(hit, "tag"),
    }
    log(f"exact evidence (windows): {res['evidence_vs_windows']['colliding_unique_chunks']}/{n_uk} "
        f"unique gold chunks")

    # --- (claim, evidence) pairs
    mp = mix.select(["claim", "chunk", "tag"]).unique()
    hit = gold.join(mp, on=["claim", "chunk"], how="inner")
    n_hit_pairs = int(hit.select(["owner_ix", "claim", "chunk"]).unique().height)
    res["pairs"] = {
        "gold_pairs": n_rows,
        "colliding_gold_pairs": n_hit_pairs,
        "colliding_gold_claims": int(hit["owner_ix"].n_unique()),
        "fraction_of_gold_pairs": round(n_hit_pairs / n_rows, 6),
        "fraction_of_gold_claims": round(hit["owner_ix"].n_unique() / n_claims, 6),
        "per_corpus": _attr(hit, "tag"),
    }
    log(f"exact pairs: {res['pairs']['colliding_gold_pairs']}/{n_rows}")
    return res


# --- near-duplicate census -----------------------------------------------------


def census(label, candidate_texts, arena_texts, spike=True):
    t = time.time()
    log(f"census {label}: {len(candidate_texts)} candidate units vs "
        f"{sum(len(v) for v in arena_texts.values())} mix units over {len(arena_texts)} groups")
    res = G.run_gate(candidate_texts, n=GATE_N, jaccard=GATE_JACCARD, kill=GATE_KILL,
                     label=label, arena_texts=arena_texts)
    out = {"result": res, "seconds": round(time.time() - t, 1)}
    log(f"census {label}: verdict {res['verdict']}  max_fraction {res['max_fraction']}  "
        f"best_jaccard {res['candidate_vs_arena'].get('best_jaccard')}  ({out['seconds']}s)")
    if spike:
        sp = G.spike_control(candidate_texts[:SPIKE_SAMPLE], arena_texts, n=GATE_N,
                             jaccard=GATE_JACCARD, k=10, label=f"{label}_spike")
        out["spike_control"] = sp
        log(f"census {label}: spike control {sp}")
    return out


# --- provenance ----------------------------------------------------------------


def provenance(gold_full_n):
    """What identifier fields actually exist on each side, and their collisions."""
    tp = pl.read_parquet(GOLD_PAIRS)
    gs = pl.read_parquet(GOLD_SRC).with_row_index("owner")

    gold_fields = {
        "R7-H51_teacher_pairs.parquet": list(tp.columns),
        "golden_grounding_evidence_verified.parquet": list(gs.columns) + ["owner (row index)"],
    }
    gold_ids = {
        "trace_id": sorted(set(gs["trace_id"].to_list())),
        "user_id": sorted({u for u in gs["user_id"].to_list() if u}),
    }

    train_fields, train_id_values = {}, {}
    zips = {
        "ragtruth_en": ("dataset-ragtruth.zip", "__train.parquet", None),
        "ragtruth_translated": ("dataset-ragtruth-translated.zip", "__train.parquet", None),
        "halueval": ("dataset-halueval.zip", ".parquet", None),
        "psiloqa": ("dataset-psiloqa.zip", "__train.parquet", None),
        "vitaminc": ("dataset-vitaminc.zip", "__train.parquet", None),
        "tabfact": ("dataset-tabfact.zip", "__train.parquet", None),
    }
    for name, (zn, suffix, _) in zips.items():
        z = zipfile.ZipFile(DATA / zn)
        hits = [x for x in z.namelist() if x.endswith(suffix)]
        if not hits:
            train_fields[name] = []
            continue
        cols = {}
        for h in hits:
            df = pl.read_parquet(io.BytesIO(z.read(h)))
            cols[h] = list(df.columns)
            for c in df.columns:
                if c.lower() in {"id", "unique_id", "case_id", "wiki_revision_id", "page",
                                 "fever_id", "table_id", "wiki_title", "wiki_url", "doc_id"}:
                    train_id_values.setdefault(f"{name}.{c}", set()).update(
                        df[c].cast(pl.Utf8).drop_nulls().to_list())
        train_fields[name] = cols
    for fname, group, *_ in H150.LANES:
        df = pl.read_parquet(HERE / fname)
        train_fields[group] = list(df.columns)
        for c in ("doc_id", "source", "row_key"):
            if c in df.columns:
                train_id_values.setdefault(f"{group}.{c}", set()).update(
                    df[c].cast(pl.Utf8).drop_nulls().to_list())

    collisions = {}
    for gname, gvals in gold_ids.items():
        gset = set(gvals)
        for tname, tvals in train_id_values.items():
            inter = gset & tvals
            if inter:
                collisions[f"{gname} x {tname}"] = {"n": len(inter), "examples": sorted(inter)[:5]}

    return {
        "gold_identifier_fields": gold_fields,
        "gold_identifier_cardinality": {
            "trace_id": len(gold_ids["trace_id"]),
            "user_id": len(gold_ids["user_id"]),
            "owner": int(gs.height),
            "gold_full_claims": gold_full_n,
        },
        "training_identifier_fields": train_fields,
        "training_identifier_value_sets": {k: len(v) for k, v in sorted(train_id_values.items())},
        "id_join_possible": False,
        "id_join_note": (
            "gold_full carries {owner (row index), trace_id, user_id, lang} and NO document / "
            "page / revision / corpus identifier. The training corpora carry their own "
            "namespaces (ragtruth id, psiloqa id+wiki_title+wiki_url, vitaminc "
            "unique_id/case_id/wiki_revision_id/page/FEVER_id, tabfact table_id, lane doc_id). "
            "The two namespaces do not intersect by construction, so a provenance JOIN - the "
            "channel that caught VitaminC - CANNOT be computed for gold_full. Reported as a "
            "measured absence, not substituted with a proxy. What IS executable is the value-set "
            "intersection below and the substring scan for gold trace/user ids in mix text."),
        "id_value_collisions": collisions,
        "gold_ids": gold_ids,
    }


def id_substring_scan(mix, prov):
    """Do gold trace/user ids appear verbatim anywhere in the mix text?"""
    needles = prov["gold_ids"]["trace_id"] + prov["gold_ids"]["user_id"]
    out = {}
    for col in ("claim", "chunk"):
        m = mix[col].str.contains_any(needles)
        out[col] = int(m.sum())
    log(f"id substring scan: {out} (needles {len(needles)})")
    return {"needles": len(needles), "mix_rows_containing_a_gold_id": out}


# --- main ----------------------------------------------------------------------


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    res = {
        "audit": "R20 gold_full vs training-mix split audit",
        "question": "is the in-domain held-out set gold_full disjoint from the assembled training mix?",
        "instrument": {
            "exact": "polars set / join equality on claim, evidence and (claim, evidence)",
            "near_duplicate": (f"provenance_gate.run_gate, R14-H136 ruling-2 form: {GATE_N}-gram, "
                               f"Jaccard >= {GATE_JACCARD}, bidirectional, KILL at {GATE_KILL:.0%}, "
                               "spike control; the TRAINING MIX replaces the arena side so the "
                               "per-bucket breakdown attributes hits to a DANN group"),
            "thresholds_read_from": "R19_supply_gates.py",
        },
        "sides": {},
    }

    gold, n_claims, y = assemble_gold()
    mix, n_mix = assemble_mix()
    res["sides"] = {
        "gold_full": {
            "loader": "R10-H108_lane.gold_full()",
            "source": str(GOLD_PAIRS.relative_to(ROOT)),
            "claims": n_claims,
            "pairs": gold.height,
            "unique_claim_strings": int(gold["claim"].n_unique()),
            "unique_chunks": int(gold["chunk"].n_unique()),
            "label_base_rate": round(float(np.mean(y)), 4),
            "served_unit_chars": SERVE_CHARS,
            "max_gold_chunk_chars": int(gold["chunk"].str.len_chars().max()),
        },
        "training_mix": {
            "loader": "R16-H142_G1_arm.untruncated_evidence() + R10-H108_lane.public_train() "
                      "+ R18-H150_arm_run.LANES",
            "rows": n_mix,
            "expected_rows": H150.EXPECTED_MIX_ROWS,
            "reproduces_expected": n_mix == H150.EXPECTED_MIX_ROWS,
            "clean_rows": H150.EXPECTED_CLEAN_ROWS,
            "groups": list(H150.EXPECTED_GROUPS),
            "rows_per_group": {r["tag"]: int(r["len"]) for r in mix.group_by("tag").len().sort("tag").iter_rows(named=True)},
            "unique_claim_strings": int(mix["claim"].n_unique()),
            "unique_chunks": int(mix["chunk"].n_unique()),
        },
    }
    OUT.write_text(json.dumps(res, indent=2))
    log("=== STAGE ASSEMBLE DONE ===")

    mixwin = mix_windows(mix)
    res["sides"]["training_mix"]["windows"] = int(mixwin.height)
    res["sides"]["training_mix"]["unique_windows"] = int(mixwin["window"].n_unique())

    res["exact_match"] = exact_channels(gold, mix, mixwin, n_claims)
    OUT.write_text(json.dumps(res, indent=2))
    log("=== STAGE EXACT DONE ===")

    prov = provenance(n_claims)
    prov["id_substring_scan"] = id_substring_scan(mix, prov)
    prov.pop("gold_ids")
    res["provenance"] = prov
    OUT.write_text(json.dumps(res, indent=2))
    log("=== STAGE PROVENANCE DONE ===")

    # near-duplicate census - claims channel
    arena_claims = {t[0]: sorted({c for c in g["claim"].to_list() if c and c.strip()})
                    for t, g in mix.group_by("tag")}
    gold_claims = sorted({c for c in gold["claim"].to_list() if c and c.strip()})
    res["near_duplicate"] = {"claims": census("goldfull_claims", gold_claims, arena_claims)}
    OUT.write_text(json.dumps(res, indent=2))
    del arena_claims
    log("=== STAGE CENSUS CLAIMS DONE ===")

    # near-duplicate census - evidence channel, mix side as WINDOWS (what was shown)
    arena_ev = {t[0]: sorted({w for w in g["window"].to_list() if w and w.strip()})
                for t, g in mixwin.group_by("tag")}
    gold_ev = sorted({c for c in gold["chunk"].to_list() if c and c.strip()})
    res["near_duplicate"]["evidence"] = census("goldfull_evidence", gold_ev, arena_ev)
    OUT.write_text(json.dumps(res, indent=2))
    log("=== STAGE CENSUS EVIDENCE DONE ===")

    # verdict material - measured only, no adjudication
    ex = res["exact_match"]
    worst_exact = max(ex["claims"]["fraction_of_gold_claims"],
                      ex["evidence_vs_raw_chunks"]["fraction_of_gold_claims"],
                      ex["evidence_vs_windows"]["fraction_of_gold_claims"],
                      ex["pairs"]["fraction_of_gold_claims"])
    res["summary"] = {
        "exact_worst_fraction_of_gold_claims": worst_exact,
        "near_duplicate_claims_max_fraction": res["near_duplicate"]["claims"]["result"]["max_fraction"],
        "near_duplicate_claims_verdict": res["near_duplicate"]["claims"]["result"]["verdict"],
        "near_duplicate_evidence_max_fraction": res["near_duplicate"]["evidence"]["result"]["max_fraction"],
        "near_duplicate_evidence_verdict": res["near_duplicate"]["evidence"]["result"]["verdict"],
        "spike_controls_pass": bool(res["near_duplicate"]["claims"]["spike_control"]["passes"]
                                    and res["near_duplicate"]["evidence"]["spike_control"]["passes"]),
        "provenance_id_join_possible": False,
        "provenance_id_value_collisions": len(res["provenance"]["id_value_collisions"]),
    }
    OUT.write_text(json.dumps(res, indent=2))
    log(f"summary: {json.dumps(res['summary'])}")
    log(f"banked -> {OUT}")
    log("=== AUDIT DONE ===")


if __name__ == "__main__":
    main()
