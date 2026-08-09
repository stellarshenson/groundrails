"""R15-B6 evidence-conditioning gate (R15-H139) == R15-B5 arm 3.

Every negative in the entire training corpus is manufactured by editing the
CLAIM. Whether the shipped model conditions on evidence numerals at all outside
the verbatim-copy route is unmeasured - and it is the premise of every lane in
the register.

Held-out TabFact, frozen H105 draw 1. The claim is byte-identical across a pair
INCLUDING its numeral; the EVIDENCE cell is edited instead.

  Clause 1 (the target)  - derived claim, correct vs perturbed operand cell.
                           LICENSE if AUROC <= 0.60.
  Clause 2 (the discriminator) - the same edit on a cell the claim quotes
                           VERBATIM. The copy detector must catch this.
                           NO-READ / ESCALATE if AUROC <= 0.60 here too - the
                           model would not be reading evidence numerals at all
                           and the whole field needs re-scoping.
  Clause 3 (CPU, free)   - certify positive and negative claim strings are
                           byte-identical; any inequality is a build bug.

Origin symmetry is enforced: half the pairs are built from the EDITED table with
the original as the negative, so "which table is pristine" cannot carry the read.

Run: CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 uv run python <this>
"""

import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
RESULT = HERE / "R15_gate_B6_result.json"
SAMPLE = HERE / "R15_gate_B6_pairs.parquet"

CKPT = "R9-H105-mmbert-dann-clean"
SEED = 20260812
N_PER_ARM = 600
BAR = 0.60
TOTAL_WORDS = ("total", "totals", "sum", "overall", "all", "aggregate")


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def digits(s):
    return sum(ch.isdigit() for ch in s)


def build(C, rng):
    caps, tbls, tids = C.held_tabfact()
    print(f"held-out tables: {len(tbls)}", flush=True)
    order = [int(o) for _ in range(3) for o in rng.permutation(len(tbls))]

    out = {"derived": [], "verbatim": []}
    seen = {"derived": set(), "verbatim": set()}
    total_row_hits, total_row_seen = 0, 0
    for oi in order:
        if all(len(out[a]) >= N_PER_ARM for a in out):
            break
        hdr, body = C.parse(tbls[oi])
        if hdr is None:
            continue
        cand = []
        for ci in range(1, len(hdr)):
            vals = [(ri, C.as_num(r[ci])) for ri, r in enumerate(body)]
            vals = [(ri, v) for ri, v in vals if v is not None]
            if len(vals) >= 4 and len({v for _, v in vals}) >= 4:
                cand.append((ci, vals))
        if not cand:
            continue
        ci, vals = cand[int(rng.integers(len(cand)))]
        col = hdr[ci] or f"column {ci}"
        if any((not body[ri][0]) or C.as_num(body[ri][0]) is not None for ri, _ in vals):
            continue
        pick = [int(p) for p in rng.permutation(len(vals))[:2]]
        (ri_a, vi), (ri_b, vj) = vals[pick[0]], vals[pick[1]]
        ka, kb = body[ri_a][0].strip(), body[ri_b][0].strip()
        if not ka or not kb or ka == kb:
            continue
        cell = body[ri_a][ci].strip()
        # replacement: same digit length, drawn from the column's own values
        alts = [C.fmt(v) for _, v in vals
                if abs(v - vi) > 1e-9 and digits(C.fmt(v)) == digits(cell)]
        alts = [a for a in alts if a != cell]
        if not alts:
            continue
        repl = alts[int(rng.integers(len(alts)))]
        vi2 = C.as_num(repl)
        if vi2 is None or abs(vi2 - vi) < 1e-9:
            continue

        body2 = [list(r) for r in body]
        body2[ri_a][ci] = repl
        ev_o = C.serialize(caps[oi], hdr, body)
        ev_e = C.serialize(caps[oi], hdr, body2)
        flip = bool(rng.integers(2))  # origin symmetry
        total_row_seen += 1
        has_total = any(str(r[0]).strip().lower() in TOTAL_WORDS for r in body)
        total_row_hits += int(has_total)

        # ---- clause 1: derived claim, byte-identical, evidence edited --------
        if len(out["derived"]) < N_PER_ARM:
            s_o, s_e = C.fmt(vi + vj), C.fmt(vi2 + vj)
            if s_o != s_e and all(x not in ev_o and x not in ev_e for x in (s_o, s_e)):
                V = s_e if flip else s_o
                claim = f"The combined {col} of {ka} and {kb} is {V}."
                key = (tids[oi], col, ka, kb, V)
                if key not in seen["derived"]:
                    seen["derived"].add(key)
                    out["derived"].append({
                        "arm": "derived", "table_id": tids[oi], "column": col, "claim": claim,
                        "origin": "edited" if flip else "original",
                        "ev_pos": ev_e if flip else ev_o, "ev_neg": ev_o if flip else ev_e,
                        "v_asserted": V, "cell_orig": cell, "cell_repl": repl,
                        "table_has_total_row": has_total})

        # ---- clause 2: verbatim claim, same edit on the quoted cell ---------
        if len(out["verbatim"]) < N_PER_ARM:
            V = repl if flip else cell
            other = cell if flip else repl
            ev_pos = ev_e if flip else ev_o
            ev_neg = ev_o if flip else ev_e
            if V != other and V in ev_pos and V not in ev_neg:
                claim = f"The {col} of {ka} is {V}."
                key = (tids[oi], col, ka, V)
                if key not in seen["verbatim"]:
                    seen["verbatim"].add(key)
                    out["verbatim"].append({
                        "arm": "verbatim", "table_id": tids[oi], "column": col, "claim": claim,
                        "origin": "edited" if flip else "original",
                        "ev_pos": ev_pos, "ev_neg": ev_neg,
                        "v_asserted": V, "cell_orig": cell, "cell_repl": repl,
                        "table_has_total_row": has_total})
    share_total = total_row_hits / max(total_row_seen, 1)
    return out, share_total


def main():
    import torch

    C = _mod("c", "R15_gate_common.py")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    built, share_total = build(C, np.random.default_rng(SEED))
    for a in built:
        print(f"  {a:10s} {len(built[a])}", flush=True)
    items = built["derived"] + built["verbatim"]
    if len(built["derived"]) < 100 or len(built["verbatim"]) < 100:
        raise SystemExit("too few evidence-side pairs constructible to adjudicate")

    # clause 3 - the claim string is one object, used for both members
    byte_identical = all(isinstance(p["claim"], str) for p in items)

    n = len(items)
    claims = [p["claim"] for p in items] * 2
    evs = [p["ev_pos"] for p in items] + [p["ev_neg"] for p in items]
    tok, trunk, head = C.load_ckpt(CKPT)
    s = C.score(tok, trunk, head, claims, evs)
    del trunk, head
    torch.cuda.empty_cache()
    sp, sn = s[:n], s[n:]

    per_arm = {}
    for a in ("derived", "verbatim"):
        m = np.array([p["arm"] == a for p in items])
        P, N = sp[m], sn[m]
        sub = {
            "n": int(m.sum()),
            "mean_pos": round(float(P.mean()), 5),
            "mean_neg": round(float(N.mean()), 5),
            "auroc_pos_vs_neg": round(C.auroc(P, N), 4),
            "frac_pos_higher": round(float((P > N).mean()), 4),
            "distinct_tables": len({p["table_id"] for p, mm in zip(items, m) if mm}),
        }
        for orig in ("original", "edited"):
            mo = np.array([p["arm"] == a and p["origin"] == orig for p in items])
            if mo.sum() >= 30:
                sub[f"auroc_origin_{orig}_is_positive"] = round(C.auroc(sp[mo], sn[mo]), 4)
                sub[f"n_origin_{orig}"] = int(mo.sum())
        per_arm[a] = sub
        print(a, json.dumps(sub), flush=True)

    pl.DataFrame([{k: v for k, v in p.items() if not k.startswith("ev_")} for p in items]
                 ).with_columns([pl.Series("score_pos", sp), pl.Series("score_neg", sn)]
                                ).write_parquet(SAMPLE)

    a1 = per_arm["derived"]["auroc_pos_vs_neg"]
    a2 = per_arm["verbatim"]["auroc_pos_vs_neg"]
    if a2 <= BAR:
        verdict = "NO-READ / ESCALATE"
        clause = (f"clause 2 fired - verbatim-evidence-edit AUROC {a2:.4f} <= {BAR}: the model is "
                  "not reading evidence numerals at all; the R14 diagnosis is too narrow and the "
                  "field needs re-scoping before any lane is built")
    elif a1 <= BAR:
        verdict = "LICENSE"
        clause = (f"clause 1 licenses - evidence-side derived AUROC {a1:.4f} <= {BAR} while the "
                  f"verbatim discriminator reads {a2:.4f} > {BAR}: the model reads evidence "
                  "numerals on the copy route but does not condition on them for derived claims")
    else:
        verdict = "KILL"
        clause = (f"clause 1 fired - evidence-side derived AUROC {a1:.4f} > {BAR}: the model "
                  "already conditions on evidence numerals for derived claims and H139 has "
                  "nothing to install")

    res = {
        "gate": "R15-B6 evidence-conditioning gate (R15-H139) == R15-B5 arm 3",
        "model": str(C.MODELS / CKPT),
        "data": "TabFact test+validation, table_id-disjoint from every train split; zero arena, "
                "zero gold",
        "implementation_choices": [
            "Both members of a pair are rendered by the SAME serializer from the same header and "
            "body, so the only difference between the two evidence strings is the edited cell.",
            "Origin symmetry is enforced pair by pair with a fair coin: half the pairs take the "
            "EDITED table as the positive (the asserted value is computed from it) and the "
            "original as the negative. Per-origin AUROCs are reported so a pristine-table cue "
            "would be visible.",
            "The replacement is drawn from the column's own empirical values at equal digit "
            "count, per L3-C2's construction.",
            "The derived operator is the two-cell sum, which is the operator the banked H133 / "
            "P1 baselines are defined on; the gate measures evidence conditioning, not operator "
            "coverage.",
            "Clause 1 requires both the original and the edited derived values ABSENT from both "
            "serializations, so no verbatim-copy route is open on either member.",
        ],
        "seed": SEED, "n_per_arm_target": N_PER_ARM,
        "clause_3_claim_strings_byte_identical": bool(byte_identical),
        "clause_3_note": "the claim is a single Python string scored against two evidence strings, "
                         "so byte-identity holds by construction",
        "share_of_edited_tables_carrying_a_total_row": round(share_total, 4),
        "arms": per_arm,
        "bar": f"LICENSE if clause 1 AUROC <= {BAR}; NO-READ / ESCALATE if clause 2 AUROC <= {BAR} "
               "as well",
        "verdict": verdict,
        "clause_fired": clause,
        "gates_downstream": "R15-H139's registration (its own ~13 GPU-h arm, or a sub-block of "
                            "A4's SECOND build); on NO-READ the whole R15 field is re-scoped",
        "sample": SAMPLE.name,
    }
    RESULT.write_text(json.dumps(res, indent=2))
    print("\n" + "=" * 88)
    print(f"  clause 1 derived  AUROC {a1:.4f} (LICENSE at <= {BAR})")
    print(f"  clause 2 verbatim AUROC {a2:.4f} (NO-READ at <= {BAR})")
    print(f"\n  VERDICT: {verdict}\n  {clause}\n  -> {RESULT}", flush=True)


if __name__ == "__main__":
    main()
