"""Phase-1 re-adjudication - the C-A1 structural C1 test on EVERY loaded member.

CPU ONLY, HF_HUB_OFFLINE. Rebuilds the 760,618-row mix through the BANKED
loader (`R10-H108_lane.public_train` under `R16-H142_G1_arm.untruncated_evidence`
plus the five `R20-H174_arm_run.LANES`) and measures, per member:

  1  STRUCTURAL C1 (amendment C-A1 test 1) - the count of distinct (claim,
     evidence) pairs that carry BOTH labels, in three string forms: raw,
     evidence truncated to CFG.chunk_max_chars = 1500, and whitespace-collapsed
     case-folded on both sides. Rows covered by such pairs reported too.
  2  UNIFORM C1 ATTESTATION (amendment C-A2 tests 2 and 3) - both legs' mean
     containment, rate >= 0.90 and rate == 1.0 under ONE instrument for every
     member, `R20-H174_lane_common.containment` (banked ASCII content-token
     containment). This is the predicate-BLIND instrument; each member's own
     predicate-sensitive reading is taken from its banked report, not here.

LIVE POSITIVE CONTROL: the withdrawn poisoned `R20-H175b_qlane.parquet`, which
must read 8,986 both-label pairs over 17,972 rows.

Writes phase1_readjudication_structural.json. Measurement only - no verdicts.

Run:  CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 uv run python \
      experiments/grounding-semantic/contract/phase1_readjudication_structural.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HUB_OFFLINE"] = "1"

import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
SEM = HERE.parent
CHUNK_MAX = 1500

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[a-z0-9]+")

# tag (DANN group) -> contract member
MEMBER_OF = {
    "ragtruth_en": "ragtruth_en",
    "ragtruth_cn": "ragtruth_translated",
    "ragtruth_de": "ragtruth_translated",
    "ragtruth_es": "ragtruth_translated",
    "ragtruth_fr": "ragtruth_translated",
    "ragtruth_hu": "ragtruth_translated",
    "ragtruth_it": "ragtruth_translated",
    "ragtruth_pl": "ragtruth_translated",
    "halueval": "halueval",
    "psiloqa": "psiloqa",
    "vitaminc": "vitaminc",
    "tabfact": "tabfact",
    "quant_misbind": "quant_misbind",
    "quant_scale_unit": "quant_scale_unit",
    "frame_reject": "frame_reject",
    "attr_pool": "attr_pool",
    "path_bind": "path_bind",
}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, SEM / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm_ws(t):
    return _WS.sub(" ", t).strip().lower()


def structural(df, form):
    """Distinct (claim, evidence) pairs carrying BOTH labels, under `form`.

    Grouping is on 64-bit hashes of the two sides; every flagged group is then
    re-verified on the literal strings so a hash collision cannot manufacture a
    hit."""
    g = (
        df.group_by(["ch", "cl"])
        .agg(
            pl.len().alias("rows"),
            pl.col("label").n_unique().alias("nlab"),
        )
        .filter(pl.col("nlab") > 1)
    )
    if g.height == 0:
        return {"form": form, "both_label_pairs": 0, "rows_covered": 0,
                "verified_on_literal_strings": True}
    # re-verify on literal strings
    hit = df.join(g.select("ch", "cl"), on=["ch", "cl"], how="inner")
    v = (
        hit.group_by(["chunk_f", "claim_f"])
        .agg(pl.len().alias("rows"), pl.col("label").n_unique().alias("nlab"))
        .filter(pl.col("nlab") > 1)
    )
    return {
        "form": form,
        "both_label_pairs": int(v.height),
        "rows_covered": int(v["rows"].sum()) if v.height else 0,
        "hash_level_pairs": int(g.height),
        "verified_on_literal_strings": True,
    }


def structural_all_forms(claims, chunks, labels):
    df = pl.DataFrame(
        {"claim": claims, "chunk": chunks, "label": np.asarray(labels, dtype="int8")}
    )
    out = {}
    # raw
    d = df.with_columns(
        pl.col("chunk").alias("chunk_f"), pl.col("claim").alias("claim_f")
    ).with_columns(pl.col("chunk_f").hash().alias("ch"), pl.col("claim_f").hash().alias("cl"))
    out["raw"] = structural(d, "raw")
    # evidence truncated to 1500
    d = df.with_columns(
        pl.col("chunk").str.slice(0, CHUNK_MAX).alias("chunk_f"),
        pl.col("claim").alias("claim_f"),
    ).with_columns(pl.col("chunk_f").hash().alias("ch"), pl.col("claim_f").hash().alias("cl"))
    out["evidence_truncated_1500"] = structural(d, "evidence_truncated_1500")
    # whitespace-collapsed case-folded, both sides
    d = df.with_columns(
        pl.col("chunk").map_elements(norm_ws, return_dtype=pl.String).alias("chunk_f"),
        pl.col("claim").map_elements(norm_ws, return_dtype=pl.String).alias("claim_f"),
    ).with_columns(pl.col("chunk_f").hash().alias("ch"), pl.col("claim_f").hash().alias("cl"))
    out["whitespace_collapsed_casefolded"] = structural(
        d, "whitespace_collapsed_casefolded"
    )
    return out


def containment_legs(claims, chunks, labels, member):
    """Both legs under the banked ASCII containment instrument.

    Rows are walked in evidence-sorted order so an identical consecutive
    evidence string reuses its token set - O(1) memory, full dedup benefit."""
    idx = sorted(range(len(chunks)), key=chunks.__getitem__)
    scores = np.empty(len(claims), dtype="float64")
    last_txt, last_set = None, None
    t0 = time.time()
    for n, i in enumerate(idx):
        txt = chunks[i]
        if txt != last_txt:
            last_set = set(_WORD.findall(txt.lower()))
            last_txt = txt
        ct = set(_WORD.findall(claims[i].lower()))
        scores[i] = (len(ct & last_set) / len(ct)) if ct else 0.0
        if n and n % 100_000 == 0:
            print(f"    {member}: {n}/{len(idx)} scored "
                  f"({time.time() - t0:.0f}s)", flush=True)
    y = np.asarray(labels, dtype="int8")
    out = {}
    for name, leg in (("positive_leg", scores[y == 1]), ("negative_leg", scores[y == 0])):
        out[name] = {
            "n": int(leg.size),
            "mean": round(float(leg.mean()), 6) if leg.size else None,
            "rate_ge_0.90": round(float((leg >= 0.90).mean()), 6) if leg.size else None,
            "rate_eq_1.00": round(float((leg >= 1.0).mean()), 6) if leg.size else None,
        }
    p, n_ = out["positive_leg"], out["negative_leg"]
    out["test2_neg_strictly_below_pos_at_ge_0.90"] = (
        None if p["rate_ge_0.90"] is None or n_["rate_ge_0.90"] is None
        else bool(n_["rate_ge_0.90"] < p["rate_ge_0.90"])
    )
    out["test2_neg_strictly_below_pos_at_eq_1.00"] = (
        None if p["rate_eq_1.00"] is None or n_["rate_eq_1.00"] is None
        else bool(n_["rate_eq_1.00"] < p["rate_eq_1.00"])
    )
    out["instrument"] = "R20-H174_lane_common.containment (banked ASCII [a-z0-9]+)"
    out["instrument_class"] = "predicate-BLIND baseline, uniform across members"
    return out


def main():
    t_all = time.time()
    arm = _mod("g1arm", "R16-H142_G1_arm.py")
    H108 = arm.H108
    h174 = _mod("h174arm", "R20-H174_arm_run.py")
    h150 = _mod("h150arm", "R18-H150_arm_run.py")

    with arm.untruncated_evidence():
        claims, chunks, y, tags = H108.public_train()
    print(f"clean mix rows {len(y)} (expect {h174.EXPECTED_CLEAN_ROWS})", flush=True)
    assert len(y) == h174.EXPECTED_CLEAN_ROWS, len(y)

    for fname, group, n_rows, n_pairs, _fams in h174.LANES:
        df = pl.read_parquet(SEM / fname)
        assert len(df) == n_rows, (fname, len(df), n_rows)
        assert df["pair_id"].n_unique() == n_pairs, fname
        claims += df["claim"].to_list()
        chunks += df["chunk"].to_list()
        y = np.concatenate([y, df["label"].cast(pl.Float32).to_numpy()])
        tags += [group] * len(df)
        print(f"lane {group}: {len(df)} rows / {n_pairs} pairs", flush=True)

    assert len(y) == h174.EXPECTED_MIX_ROWS, len(y)
    print(f"MIX ROWS {len(y)} - loader reconciles", flush=True)

    y = y.astype("int8")
    members = {}
    tags_np = np.asarray(tags, dtype=object)
    member_np = np.asarray([MEMBER_OF[t] for t in tags], dtype=object)

    for m in sorted(set(member_np)):
        sel = np.flatnonzero(member_np == m)
        mc = [claims[i] for i in sel]
        mk = [chunks[i] for i in sel]
        ml = y[sel]
        print(f"member {m}: {len(sel)} rows", flush=True)
        rec = {
            "rows": int(len(sel)),
            "dann_groups": sorted(set(tags_np[sel].tolist())),
            "label_1_rows": int((ml == 1).sum()),
            "label_0_rows": int((ml == 0).sum()),
            "structural_C1": structural_all_forms(mc, mk, ml),
            "uniform_containment_C1": containment_legs(mc, mk, ml, m),
        }
        rec["distinct_pairs_raw"] = int(
            pl.DataFrame({"a": mc, "b": mk}).n_unique()
        )
        members[m] = rec
        print(f"  -> structural raw both-label pairs "
              f"{rec['structural_C1']['raw']['both_label_pairs']}", flush=True)

    # cross-member reading over the whole mix
    print("whole-mix cross-member structural", flush=True)
    whole = structural_all_forms(claims, chunks, y)

    # ------------------------------------------------------------------ #
    # LIVE POSITIVE CONTROL - the withdrawn poisoned R20-H175b qlane
    # ------------------------------------------------------------------ #
    print("positive control: R20-H175b_qlane.parquet", flush=True)
    q = pl.read_parquet(SEM / "R20-H175b_qlane.parquet")
    qc, qk = q["claim"].to_list(), q["chunk"].to_list()
    ql = q["label"].cast(pl.Int8).to_numpy()
    control = {
        "artifact": "R20-H175b_qlane.parquet",
        "rows": int(len(q)),
        "structural_C1": structural_all_forms(qc, qk, ql),
        "uniform_containment_C1": containment_legs(qc, qk, ql, "R20-H175b_qlane"),
        "expected_both_label_pairs": 8986,
        "expected_rows_covered": 17972,
    }
    raw = control["structural_C1"]["raw"]
    control["control_fires_as_registered"] = bool(
        raw["both_label_pairs"] == 8986 and raw["rows_covered"] == 17972
    )
    print(f"  control: {raw['both_label_pairs']} pairs / {raw['rows_covered']} rows "
          f"(expect 8986 / 17972) -> fires={control['control_fires_as_registered']}",
          flush=True)

    out = {
        "task": "C-A1 structural C1 test re-run on every loaded member, plus a "
                "uniform predicate-blind attestation reading for C-A2 tests 2/3",
        "contract": "docs/experiments/dataset-contract.md, amendments C-A1 and C-A2",
        "compute": "CPU only, HF_HUB_OFFLINE=1",
        "mix": {
            "loader": "R10-H108_lane.public_train under "
                      "R16-H142_G1_arm.untruncated_evidence + R20-H174_arm_run.LANES",
            "clean_rows": int(h174.EXPECTED_CLEAN_ROWS),
            "mix_rows": int(h174.EXPECTED_MIX_ROWS),
            "measured_rows": int(len(y)),
        },
        "structural_test_definition":
            "C-A1 test 1 - a distinct (claim, evidence) pair that carries BOTH "
            "labels means no function of (claim, evidence) separates the legs, "
            "so the label cannot encode grounding. Counted per member; three "
            "string forms reported.",
        "live_positive_control": control,
        "members": members,
        "whole_mix_cross_member": whole,
        "seconds": round(time.time() - t_all, 1),
        "note": "Numbers recorded, not adjudicated - the coordinator adjudicates.",
    }
    p = HERE / "phase1_readjudication_structural.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"wrote {p} in {out['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
