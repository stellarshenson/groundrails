"""R15-B5 arm 8 step 1 (CPU) - build the NATURAL-DERIVATION baseline candidates.

The lane is otherwise entirely synthetic and has no instrument that says its
arithmetic supervision transfers to derivations that occur in the wild. P3
admits VitaminC and WiCE as probes (never as lane mass): 5,837 VitaminC train
rows assert an absent number reachable by a two-operand derivation over the
evidence, against a 2.58% shuffle-control coincidence floor, and WiCE contributes
~175 real derivations over long-form web prose.

This step detects candidates and constructs the matched wrong-operand negative;
the judge pass (step 2) verifies the VitaminC leg; the read (step 3) scores them.

Detector, byte-identical to P3 section 7: a claim number is ABSENT if its
canonical form does not appear in the evidence, and DERIVABLE if it equals
a+b, a-b, b-a, a/b, b/a or either percent change for some pair among the first
40 evidence numbers.

Run: uv run python experiments/grounding-semantic/R15_gate_B5arm8_build.py
"""

import importlib.util
import io
import itertools
import json
import pathlib
import zipfile

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent
DATA = ROOT / "data" / "external" / "datasets"
OUT = HERE / "R15_gate_B5arm8_candidates.parquet"
STATS = HERE / "R15_gate_B5arm8_build.json"

SEED = 20260812
N_VITC = 500
MAX_EV_NUMS = 40


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _mod("c", "R15_gate_common.py")


def numerals(s):
    """(surface, value) for every free-standing numeral, in order."""
    out = []
    for m in C.NUM_FREE.finditer(s or ""):
        try:
            v = float(m.group(0).replace(",", ""))
        except (ValueError, OverflowError):
            continue
        if v != v or abs(v) > 1e15:
            continue
        out.append((m.group(0), v, m.start()))
    return out


def ndec(surface):
    return len(surface.split(".")[1]) if "." in surface else 0


def render(v, surface):
    """Format v the way the claim renders its own derived numeral."""
    d = ndec(surface)
    s = f"{v:,.{d}f}" if "," in surface else f"{v:.{d}f}"
    return s


OPS = {
    "sum": lambda a, b: a + b,
    "diff_ab": lambda a, b: a - b,
    "diff_ba": lambda a, b: b - a,
    "ratio_ab": lambda a, b: a / b if abs(b) > 1e-9 else None,
    "ratio_ba": lambda a, b: b / a if abs(a) > 1e-9 else None,
    "pct_ab": lambda a, b: (b - a) / a * 100 if abs(a) > 1e-9 else None,
    "pct_ba": lambda a, b: (a - b) / b * 100 if abs(b) > 1e-9 else None,
}


def match(v, target, d):
    return v is not None and abs(v) < 1e15 and round(v, d) == round(target, d)


def detect(claim, evidence):
    """First (op, a, b, surface, value) that derives an ABSENT claim numeral."""
    cl = numerals(claim)
    if not cl:
        return None
    ev_can = C.canon_set(evidence)
    absent = [(s, x, p) for s, x, p in cl if not (C.canon_set(s) & ev_can)]
    if not absent:
        return None
    ev_nums = [v for _, v, _ in numerals(evidence)][:MAX_EV_NUMS]
    if len(ev_nums) < 2:
        return None
    for surf, x, pos in absent:
        d = ndec(surf)
        for a, b in itertools.combinations(range(len(ev_nums)), 2):
            va, vb = ev_nums[a], ev_nums[b]
            for op, f in OPS.items():
                if match(f(va, vb), x, d):
                    return {"op": op, "a": va, "b": vb, "surface": surf, "value": x, "pos": pos,
                            "ev_nums": ev_nums}
    return None


def negative(det, evidence, rng):
    """Wrong-operand: the SAME operation over a DIFFERENT evidence pair, absent."""
    f = OPS[det["op"]]
    d = ndec(det["surface"])
    ev_can = C.canon_set(evidence)
    want_digits = sum(ch.isdigit() for ch in det["surface"])
    cands = []
    idx = list(itertools.combinations(range(len(det["ev_nums"])), 2))
    for i in [int(k) for k in rng.permutation(len(idx))]:
        a, b = idx[i]
        va, vb = det["ev_nums"][a], det["ev_nums"][b]
        if (va, vb) == (det["a"], det["b"]):
            continue
        v = f(va, vb)
        if v is None or v != v or abs(v) > 1e15 or round(v, d) == round(det["value"], d):
            continue
        s = render(v, det["surface"])
        if C.canon_set(s) & ev_can:
            continue
        cands.append((abs(sum(ch.isdigit() for ch in s) - want_digits), s, va, vb))
        if len(cands) >= 40:
            break
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    _, s, va, vb = cands[0]
    return {"surface_neg": s, "a_neg": va, "b_neg": vb}


def rewrite(claim, det, s_neg):
    p, q = det["pos"], det["pos"] + len(det["surface"])
    return claim[:p] + s_neg + claim[q:]


def harvest(rows, source, rng, limit=None):
    out, seen = [], set()
    for rid, claim, evidence, label in rows:
        if limit and len(out) >= limit:
            break
        ev = (evidence or "")[:C.CHUNK_MAX]
        if not claim or len(ev) < 20:
            continue
        det = detect(claim, ev)
        if det is None:
            continue
        neg = negative(det, ev, rng)
        if neg is None:
            continue
        key = (claim, det["surface"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "source": source, "row_id": str(rid), "label_raw": str(label),
            "claim_pos": claim, "claim_neg": rewrite(claim, det, neg["surface_neg"]),
            "evidence": ev, "op": det["op"], "a": det["a"], "b": det["b"],
            "v_correct": det["surface"], "v_wrong": neg["surface_neg"],
            "a_neg": neg["a_neg"], "b_neg": neg["b_neg"],
        })
    return out


def main():
    rng = np.random.default_rng(SEED)

    z = zipfile.ZipFile(DATA / "dataset-vitaminc.zip")
    v = pl.read_parquet(io.BytesIO(z.read("tals__vitaminc__train.parquet")))
    order = [int(o) for o in rng.permutation(len(v))]
    vr = [(v["unique_id"][i], v["claim"][i], v["evidence"][i], v["label"][i]) for i in order]
    vit = harvest(vr, "vitaminc", rng, limit=N_VITC)
    print(f"vitaminc candidates: {len(vit)} (target {N_VITC})", flush=True)

    wice = []
    for split in ("claim_train", "subclaim_train"):
        rows = []
        with open(DATA / "wice" / f"{split}.jsonl") as fh:
            for i, line in enumerate(fh):
                d = json.loads(line)
                rows.append((f"{split}:{i}", d["claim"], " ".join(d["evidence"]), d["label"]))
        got = harvest(rows, f"wice_{split}", rng)
        print(f"{split} candidates: {len(got)}", flush=True)
        wice += got

    items = vit + wice
    if not items:
        raise SystemExit("no natural derivation candidates constructible")
    pl.DataFrame(items).write_parquet(OUT)

    stats = {
        "step": "R15-B5 arm 8 step 1 - natural-derivation candidate build",
        "seed": SEED,
        "detector": "P3 section 7 - absent claim numeral equal to a+b, a-b, b-a, a/b, b/a or "
                    f"either percent change over the first {MAX_EV_NUMS} evidence numbers, at the "
                    "claim numeral's own decimal precision",
        "negative": "the SAME operation over a DIFFERENT evidence pair, rendered in the claim's "
                    "own surface, absent from the evidence, digit-length-matched where "
                    "constructible; the claim is byte-identical outside the numeral",
        "n_vitaminc": len(vit), "n_wice": len(wice), "n_total": len(items),
        "wice_caveat": "partially_supported dominates WiCE's numeric slice, so the binary "
                       "collapse is lossiest exactly where the derivations live (P3 section 7)",
        "coincidence_floor": "P3's shuffle control puts the false-positive floor at 2.58% against "
                             "a 5.10% real rate - roughly half of every detected derivation is "
                             "arithmetic coincidence, which is what the judge pass removes",
        "out": OUT.name,
    }
    STATS.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
