"""R20 gate 5 (hotpotqa G0b) - composed-probe baseline on the banked flagship.

Recipe from `docs/experiments/briefs/R20-fanout-hotpotqa-composition-hypotheses.md`
(HYP-1 kill-gate, line 52): build the 1,000-item synthetic bridge + conjunction
composed probes on CPU and read the banked flagship checkpoint on them.
KILL the composed-supply arm if the baseline leg already reads >= 0.70 AUROC.

Construction - TabFact two-table join, held-out tables only (table_id-disjoint
from every train split, `R15_gate_common.held_tabfact`), so the probe is
document-disjoint from training. HotpotQA train and HoVer are NEVER touched.

  bridge      two tables sharing a row key K. The claim elides K: "The <c1> of
              the <key header> whose <c2> is <v2> is <v1>." - c1/v1 live only in
              document A, c2/v2 only in document B, so no single document
              decides it
  conjunction "<K> has <c1> of <v1> and <c2> of <v2>." - one conjunct per
              document

  negatives   PRIMARY leg: the substituted value is another row's value from the
              SAME table, so every surface element of the negative is present in
              the bag and only the composition is wrong (the composed negative
              the arm would have to detect). SECONDARY leg: the substituted value
              is absent from the bag - the easy construction, read for contrast

Evidence is the untruncated serialised table per document, windowed 1,500/750,
max over all windows of both documents - the shipped read.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20-G0b_composed_probes.py
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import collections
import importlib.util
import json
import pathlib
import time

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
PROBES = HERE / "R20-G0b_composed_probes.parquet"
OUT = HERE / "R20-G0b_composed_probe_baseline.json"
CKPT = "R18-H150-arm-draw1"
N_PAIRS_PER_FAMILY = 250
SEED = 20260816
WIN, STRIDE = 1500, 750
KILL_AT = 0.70


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def windows(chunk):
    n = len(chunk)
    if n <= WIN:
        return [chunk]
    starts = list(range(0, n - WIN + 1, STRIDE))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    return [chunk[s : s + WIN] for s in starts]


def serialize_full(cap, txt):
    """The probes' evidence string, UNTRUNCATED (R15_gate_common.serialize without
    its CHUNK_MAX cut - the read windows it instead)."""
    return f"{cap}\n{txt}".replace("\r\n", "\n").replace("#", " | ")


def build(C, rng):
    caps, tbls, tids = C.held_tabfact()
    parsed = {}
    key2tab = collections.defaultdict(set)
    for i, t in enumerate(tbls):
        hdr, body = C.parse(t)
        if hdr is None or len(body) < 3:
            continue
        parsed[i] = (hdr, body)
        for r in body:
            k = r[0].strip()
            if len(k) >= 4 and C.as_num(k) is None:
                key2tab[k.lower()].add(i)
    shared = sorted((k, sorted(v)) for k, v in key2tab.items() if len(v) >= 2)
    order = rng.permutation(len(shared))

    rows = []
    counts = {"bridge": 0, "conjunction": 0}
    used_pairs = set()
    for oi in order:
        if all(counts[f] >= N_PAIRS_PER_FAMILY for f in counts):
            break
        key, tabs = shared[int(oi)]
        ta, tb = tabs[0], tabs[1]
        if (ta, tb) in used_pairs:
            continue
        hdr_a, body_a = parsed[ta]
        hdr_b, body_b = parsed[tb]
        ev_a, ev_b = serialize_full(caps[ta], tbls[ta]), serialize_full(caps[tb], tbls[tb])
        ra = next((r for r in body_a if r[0].strip().lower() == key), None)
        rb = next((r for r in body_b if r[0].strip().lower() == key), None)
        if ra is None or rb is None:
            continue

        def pick_col(hdr, row, body, ev):
            cand = []
            for ci in range(1, len(hdr)):
                v = row[ci].strip()
                if not v or not hdr[ci].strip() or v not in ev:
                    continue
                alts = {r[ci].strip() for r in body
                        if r[0].strip().lower() != key and r[ci].strip()
                        and r[ci].strip() != v and r[ci].strip() in ev}
                if alts:
                    cand.append((ci, v, sorted(alts)))
            return cand[int(rng.integers(0, len(cand)))] if cand else None

        pa, pb = pick_col(hdr_a, ra, body_a, ev_a), pick_col(hdr_b, rb, body_b, ev_b)
        if pa is None or pb is None:
            continue
        ci_a, v1, alts_a = pa
        ci_b, v2, _alts_b = pb
        if v1 == v2:
            continue
        c1, c2 = hdr_a[ci_a].strip(), hdr_b[ci_b].strip()
        key_hdr = hdr_a[0].strip() or "entry"
        v1_alt = alts_a[int(rng.integers(0, len(alts_a)))]
        # absent variant: a numeric-looking string not present in either document
        v1_absent = None
        base = C.as_num(v1)
        if base is not None:
            for d in (7, 13, 29, 53, 101):
                cand = C.fmt(base + d)
                if cand not in ev_a and cand not in ev_b:
                    v1_absent = cand
                    break
        if v1_absent is None:
            v1_absent = f"{v1}x"

        fam = "bridge" if counts["bridge"] < N_PAIRS_PER_FAMILY else "conjunction"
        if counts[fam] >= N_PAIRS_PER_FAMILY:
            continue
        if fam == "bridge":
            tpl = (f"The {c1} of the {key_hdr} whose {c2} is {{v}} is {{w}}.")
            pos = tpl.format(v=v2, w=v1)
            neg = tpl.format(v=v2, w=v1_alt)
            neg_abs = tpl.format(v=v2, w=v1_absent)
        else:
            tpl = f"{ra[0].strip()} has {c1} of {{w}} and {c2} of {v2}."
            pos = tpl.format(w=v1)
            neg = tpl.format(w=v1_alt)
            neg_abs = tpl.format(w=v1_absent)
        counts[fam] += 1
        used_pairs.add((ta, tb))
        pid = len(rows) // 3
        for claim, lab, leg in ((pos, 1, "positive"), (neg, 0, "negative_present"),
                                (neg_abs, 0, "negative_absent")):
            rows.append({"pair_id": pid, "family": fam, "leg": leg, "label": lab,
                         "claim": claim, "doc_a": ev_a, "doc_b": ev_b,
                         "table_id_a": tids[ta], "table_id_b": tids[tb],
                         "join_key": key})
    return pl.DataFrame(rows), counts


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    if PROBES.exists():
        df = pl.read_parquet(PROBES)
        print(f"probes already banked: {df.height} rows", flush=True)
    else:
        t0 = time.time()
        df, counts = build(C, np.random.default_rng(SEED))
        df.write_parquet(PROBES)
        print(f"built {df.height} rows {counts} in {time.time() - t0:.0f}s", flush=True)

    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    flat_c, flat_w, owner = [], [], []
    for i, r in enumerate(df.iter_rows(named=True)):
        for doc in (r["doc_a"], r["doc_b"]):
            for w in windows(doc):
                flat_c.append(r["claim"])
                flat_w.append(w)
                owner.append(i)
    owner = np.array(owner)
    tok, trunk, head = C.load_ckpt(CKPT)
    s_pair = C.score(tok, trunk, head, flat_c, flat_w)
    s = np.array([s_pair[owner == i].max() for i in range(df.height)])
    del trunk, head
    torch.cuda.empty_cache()

    df = df.with_columns(pl.Series("score", s))
    df.write_parquet(PROBES)

    def leg_auroc(fam, negleg):
        d = df.filter((pl.col("family") == fam) if fam else pl.lit(True))
        pos = d.filter(pl.col("leg") == "positive")["score"].to_numpy()
        neg = d.filter(pl.col("leg") == negleg)["score"].to_numpy()
        return round(C.auroc(pos, neg), 4), len(pos), len(neg)

    res = {}
    for negleg in ("negative_present", "negative_absent"):
        res[negleg] = {}
        for fam in (None, "bridge", "conjunction"):
            a, npos, nneg = leg_auroc(fam, negleg)
            res[negleg][fam or "pooled"] = {"auroc": a, "n_pos": npos, "n_neg": nneg}

    primary = res["negative_present"]["pooled"]["auroc"]
    verdict = "KILL" if primary >= KILL_AT else "PASS"
    payload = {
        "gate": "R20 gate 5 (hotpotqa G0b) - composed-probe baseline leg",
        "recipe_summary": ("TabFact two-table-join generator over HELD-OUT tables only "
                           f"({N_PAIRS_PER_FAMILY} bridge + {N_PAIRS_PER_FAMILY} "
                           "conjunction pairs = 1,000 probe items on the primary leg); "
                           "each item is a 2-document bag windowed 1,500/750, claim scored "
                           "against every window, max over windows; banked flagship draw 1 "
                           "read frozen"),
        "checkpoint": CKPT,
        "n_rows": df.height,
        "n_pairs": int(df["pair_id"].n_unique()),
        "n_probe_items_primary": int(df.filter(pl.col("leg") != "negative_absent").height),
        "contamination": ("TabFact held-out tables only (table_id-disjoint from every train "
                          "split); HotpotQA train and HoVer never read"),
        "auroc": res,
        "primary_leg": "negative_present (every surface element of the negative is in the "
                       "bag; only the composition is wrong)",
        "primary_auroc": primary,
        "threshold": f"KILL the composed-supply arm if the baseline leg reads >= {KILL_AT}",
        "verdict": verdict,
        "timestamp": time.strftime("%F %T"),
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print(f"primary {primary} -> {verdict}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
