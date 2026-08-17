"""C6 supplement for `quant_misbind` - the association the ASSEMBLED MIX carries
on this member's pair key, and the member-vs-mix key sharing that would create it.

C6's test is "for each pair, measure overlap between the claim and whatever the
training mix associates with that pair's key".  The pair key of this member is
the evidence chunk - the two legs share it byte-identically.  So the measurement
is: does any OTHER member of the assembled mix carry the same key, and does what
it associates with that key separate this member's classes?

The mix is rebuilt through the BANKED loader (`R10-H108_lane.public_train`) plus
the R20-H174 `LANES` tuple - nothing is re-implemented.

CPU ONLY.  Run:
  CUDA_VISIBLE_DEVICES= uv run python \
    experiments/grounding-semantic/contract/quant_misbind_mix_assoc.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import collections
import importlib.util
import json
import pathlib
import re
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
GS = HERE.parent
ROOT = GS.parent.parent
CHUNK_MAX = 1500
_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def norm_ws(s):
    return _WS.sub(" ", s).strip().lower()


def tok(t):
    return _WORD.findall(t.lower())


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def auroc(y, s):
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    if not pos.size or not neg.size:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=float)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[y == 1].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    t0 = time.time()
    H108 = _mod("h108", GS / "R10-H108_lane.py")
    H174 = _mod("h174", GS / "R20-H174_arm_run.py")

    print("building the clean public mix through the banked loader...", flush=True)
    claims, chunks, y, tags = H108.public_train()
    print(f"  clean mix: {len(claims)} rows", flush=True)

    lane_df = pl.read_parquet(GS / "R17-H146_lane.parquet")
    other = collections.defaultdict(lambda: {"claims": [], "chunks": [], "y": []})
    for c, k, lab, tg in zip(claims, chunks, y.tolist(), tags):
        other[tg]["claims"].append(c)
        other[tg]["chunks"].append(k)
        other[tg]["y"].append(lab)

    for fname, group, *_ in H174.LANES:
        if group == "quant_misbind":
            continue
        d = pl.read_parquet(GS / fname)
        other[group]["claims"] += d["claim"].to_list()
        other[group]["chunks"] += d["chunk"].to_list()
        other[group]["y"] += d["label"].cast(pl.Float32).to_list()
        print(f"  lane {group}: {d.height} rows", flush=True)

    total = sum(len(v["claims"]) for v in other.values()) + lane_df.height
    print(f"  assembled mix (incl. this member): {total} rows, "
          f"{len(other) + 1} groups", flush=True)

    lane_chunks_raw = set(lane_df["chunk"].to_list())
    lane_chunks_trunc = {c[:CHUNK_MAX] for c in lane_chunks_raw}
    lane_chunks_norm = {norm_ws(c) for c in lane_chunks_raw}
    lane_claims_norm = {norm_ws(c) for c in lane_df["claim"].to_list()}

    per_group = {}
    key_owner = {}
    for g, v in other.items():
        ck_raw = set(v["chunks"])
        ck_tr = {c[:CHUNK_MAX] for c in v["chunks"]}
        ck_no = {norm_ws(c) for c in v["chunks"]}
        cl_no = {norm_ws(c) for c in v["claims"]}
        shared_raw = lane_chunks_raw & ck_raw
        shared_tr = lane_chunks_trunc & ck_tr
        shared_no = lane_chunks_norm & ck_no
        per_group[g] = {
            "rows": len(v["claims"]),
            "shared_evidence_raw": len(shared_raw),
            "shared_evidence_truncated_1500": len(shared_tr),
            "shared_evidence_normalised": len(shared_no),
            "shared_claims_normalised": len(lane_claims_norm & cl_no),
        }
        for c in shared_no:
            key_owner.setdefault(c, []).append(g)

    # the association itself: for every lane row whose key is shared, what label
    # does the rest of the mix attach to that key?
    assoc_rows = 0
    feat = np.full(lane_df.height, np.nan)
    if key_owner:
        by_key = collections.defaultdict(list)
        for g, v in other.items():
            for c, lab in zip(v["chunks"], v["y"]):
                k = norm_ws(c)
                if k in key_owner:
                    by_key[k].append(lab)
        for i, c in enumerate(lane_df["chunk"].to_list()):
            k = norm_ws(c)
            if k in by_key:
                feat[i] = float(np.mean(by_key[k]))
                assoc_rows += 1

    ok = ~np.isnan(feat)
    labels = lane_df["label"].to_numpy()
    a = auroc(labels[ok], feat[ok]) if ok.sum() and len(set(labels[ok].tolist())) > 1 else None

    out = {
        "clause": "C6 (mix-association supplement)",
        "mix_rows_total": total,
        "mix_groups": sorted(list(other) + ["quant_misbind"]),
        "member_rows": lane_df.height,
        "pair_key": "the evidence chunk - both legs of a pair carry it byte-identically",
        "key_sharing_with_other_mix_members": per_group,
        "keys_shared_with_any_other_member": len(key_owner),
        "mix_keyed_label_association": {
            "definition": "mean label the REST of the mix attaches to this row's "
                          "evidence key (whitespace-normalised)",
            "coverage_rows": int(assoc_rows),
            "coverage_share": round(float(assoc_rows / lane_df.height), 6),
            "auroc_vs_label": None if a is None else round(a, 6),
            "reading": "UNDEFINED when coverage is 0 - no other member of the mix "
                       "carries this member's pair key, so there is no training "
                       "association to memorise",
        },
        "within_pair_note": "any feature keyed on the pair key takes the SAME value on "
                            "both legs, so its within-pair separation is exactly 0.5 "
                            "regardless of the value it takes",
        "seconds": round(time.time() - t0, 1),
    }
    p = HERE / "quant_misbind_c6_mix_assoc.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
