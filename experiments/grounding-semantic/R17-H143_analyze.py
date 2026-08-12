"""R17-H143 Stage A - analysis, branch adjudication, result JSON.

Reads R17-H143_scores.parquet, derives the per-pair score, and writes
R17-H143_stageA_result.json.

Score derivation (per model):
  * margin extracted at the verdict word's token position -> that margin
  * verdict parsed but no margin recoverable -> +/- the model's mean |margin|
  * neither verdict word present (parse failure) -> 0.0, the neutral point of a
    logprob-margin scale

The registered wording for a parse failure is "score 0.5". 0.5 is neutral for a
probability, but NOT for a logprob margin - it sits on the GROUNDED side and
would silently credit a non-answering model. Primary AUROC therefore uses 0.0;
`pooled_auroc_literal05` records the literal-0.5 reading alongside it, so the
choice changes nothing that is not visible.
"""

import json
import pathlib
import re

import numpy as np
import polars as pl

HERE = pathlib.Path(__file__).parent
EVALSET = HERE / "R17-H143_evalset.parquet"
SCORES = HERE / "R17-H143_scores.parquet"
GATE_LOG = HERE / "R17-H143_gate.json"
OUT = HERE / "R17-H143_stageA_result.json"
SOURCE_FILE = "experiments/grounding-semantic/R14-H133_lane.v2-SUPERSEDED.parquet"

PARAMS_M = {
    "PleIAs/Baguettotron": 321,
    "PleIAs/Pleias-RAG-350M": 350,
    "PleIAs/Monad": 56,
    "HuggingFaceTB/SmolLM2-360M-Instruct": 362,
    "Qwen/Qwen3-0.6B": 596,
    "mistralai/Mistral-Small-24B-Instruct-2501": 23600,
    "speakleash/Bielik-11B-v2.3-Instruct": 11000,
    "Qwen/Qwen3-32B-FP8": 32800,
}
OVER_BUDGET = {"Qwen/Qwen3-0.6B"}
TEACHER = "mistralai/Mistral-Small-24B-Instruct-2501"
# The teacher the registered branch ladder is settled on. The registered
# Mistral-Small-24B has no cached weights (~47 GB unauthorized download); the
# author's ordered completion was a CACHED Qwen3-32B-FP8 read on GPU1 through
# the identical eval set, prompt and forced-answer discipline. It is a full
# reasoning teacher above the registered size, not a substitute ceiling, so it
# fires the registered teacher bars.
TEACHER_MEASURED = "Qwen/Qwen3-32B-FP8"
# The registered teacher's weights are NOT in the cache - its snapshot holds only
# config/tokenizer/index JSONs and two orphaned .incomplete download stubs, so it
# could not be measured. Bielik-11B is the largest cached instruct model that fits
# card 2, and is the same substitution R15 made for the same reason. It is a
# SUBSTITUTE ceiling, not the registered teacher, and cannot settle the registered
# teacher bars on its own.
TEACHER_SUBSTITUTE = "speakleash/Bielik-11B-v2.3-Instruct"

TINY_BAR, TEACHER_BAR, TEACHER_FLOOR, FAMILY_BAR = 0.70, 0.85, 0.75, 0.65
CONTROL_GATE = 0.90
ARITH_SUBSAMPLE = 100

TEACHER_SKIP_REASON = (
    "weights absent from the HF cache: the snapshot holds only config.json, "
    "params.json, tokenizer files and model.safetensors.index.json (12 KB total, "
    "0 of the 10 required shards), and the blobs directory holds two orphaned "
    ".incomplete download stubs (19.7 GB and 16.2 GB of the same blob). Measuring "
    "it needs a ~47 GB download, which this run was not authorised to start."
)

EXPR = re.compile(
    r"(-?\d[\d,]*\.?\d*)\s*([-+*/x×÷])\s*(-?\d[\d,]*\.?\d*)\s*=\s*(-?\d[\d,]*\.?\d*)"
)


def auroc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    pos, neg = labels == 1, labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n_p, n_n = pos.sum(), neg.sum()
    return float((ranks[pos].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


def derive_scores(d: pl.DataFrame, parse_fail_value: float) -> np.ndarray:
    m = d["margin"].to_numpy().astype(float)
    v = d["verdict"].to_numpy()
    fb = float(np.nanmean(np.abs(m))) if np.isfinite(m).any() else 1.0
    return np.where(
        np.isfinite(m), m,
        np.where(v == "GROUNDED", fb, np.where(v == "UNGROUNDED", -fb, parse_fail_value)),
    )


def arithmetic_rate(d: pl.DataFrame, keep: list[int]) -> tuple[float | None, int, int]:
    """Fraction of the model's own `A op B = C` expressions that recompute correctly."""
    sub = d.filter(pl.col("pair_id").is_in(keep))
    ok = tot = withexpr = 0
    for text in sub["gen_text"].to_list():
        found = EXPR.findall(text or "")
        if found:
            withexpr += 1
        for a, op, b, c in found:
            try:
                x, y, z = (float(t.replace(",", "")) for t in (a, b, c))
            except ValueError:
                continue
            if op in "*x×":
                want = x * y
            elif op == "+":
                want = x + y
            elif op == "-":
                want = x - y
            else:
                if y == 0:
                    continue
                want = x / y
            tot += 1
            ok += abs(want - z) <= max(0.011, abs(want) * 0.001)
    return (ok / tot if tot else None), tot, withexpr


def main() -> None:
    ev = pl.read_parquet(EVALSET)
    sc = pl.read_parquet(SCORES)
    gates = json.loads(GATE_LOG.read_text()) if GATE_LOG.exists() else {}

    real_ids = sorted(ev.filter(~pl.col("control"))["pair_id"].to_list())
    arith_keep = real_ids[:ARITH_SUBSAMPLE]
    fam_counts = {
        k: v for k, v in
        ev.filter(~pl.col("control"), pl.col("label") == 0)
        .group_by("neg_family").len().sort("len", descending=True).iter_rows()
    }
    top3 = list(fam_counts)[:3]
    top3_no_rel = [f for f in fam_counts if f != "r:relational"][:3]

    per_model = {}
    for name in sc["model"].unique().sort().to_list():
        d = sc.filter(pl.col("model") == name)
        real = d.filter(~pl.col("control"))
        ctrl = d.filter(pl.col("control"))
        rec: dict = {
            "name": name,
            "params_M": PARAMS_M.get(name),
            "over_budget": name in OVER_BUDGET,
            "n_scored": real.height,
        }
        if real.height:
            s = derive_scores(real, 0.0)
            rec["pooled_auroc"] = auroc(real["label"].to_numpy(), s)
            rec["pooled_auroc_literal05"] = auroc(
                real["label"].to_numpy(), derive_scores(real, 0.5)
            )
            fam = {}
            for f in fam_counts:
                # a family's AUROC = its negatives against ALL positives
                sub = real.filter((pl.col("label") == 1) | (pl.col("neg_family") == f))
                fam[f] = auroc(sub["label"].to_numpy(), derive_scores(sub, 0.0))
            rec["per_family_auroc"] = fam
            cf = {}
            for f in real["claim_form"].unique().sort().to_list():
                sub = real.filter(pl.col("claim_form") == f)
                cf[f] = auroc(sub["label"].to_numpy(), derive_scores(sub, 0.0))
            rec["per_claim_form_auroc"] = cf
            # compliance: share of free traces that never stated a verdict inside
            # the 512-token budget and had the answer forced. Does not cost the
            # pair a score - the score is read at the forced answer position.
            rec["parse_fail_rate"] = float(real["parse_fail"].mean())
            rec["margin_recovered_rate"] = float(real["margin"].is_not_null().mean())
            rec["no_answer_rate"] = float(real["verdict"].is_null().mean())
            vc = real["verdict"].value_counts().to_dicts()
            rec["verdict_counts"] = {str(r["verdict"]): r["count"] for r in vc}
            rec["latency_s_per_claim"] = float(real["latency_s"].mean())
            rec["mean_gen_tokens"] = float(real["n_gen_tokens"].mean())
            r, n_expr, n_rows = arithmetic_rate(real, arith_keep)
            rec["arithmetic_correct_rate"] = r
            rec["arithmetic_expressions_found"] = n_expr
            rec["arithmetic_rows_with_expressions"] = n_rows
            rec["arithmetic_subsample_n"] = ARITH_SUBSAMPLE
        if ctrl.height:
            rec["control_auroc"] = auroc(
                ctrl["label"].to_numpy(), derive_scores(ctrl, 0.0)
            )
            rec["control_parse_fail_rate"] = float(ctrl["parse_fail"].mean())
        g = gates.get(name, {})
        # derived here, not read from the gate file: the tiny and teacher
        # processes share that file and the last writer wins, so its entries
        # are not complete
        exempt = bool(g.get("gate_exempt", name == "PleIAs/Monad"))
        rec["gate_exempt"] = exempt
        rec["control_gate_passed"] = (
            None if rec.get("control_auroc") is None
            else bool(exempt or rec["control_auroc"] >= CONTROL_GATE)
        )
        if g.get("error"):
            rec["error"] = g["error"]
            rec["skipped"] = True
        if g.get("quant_path"):
            rec["quant_path"] = g["quant_path"]
        per_model[name] = rec

    for name in PARAMS_M:
        if name not in per_model:
            per_model[name] = {
                "name": name, "params_M": PARAMS_M[name], "skipped": True,
                "error": TEACHER_SKIP_REASON if name == TEACHER
                else gates.get(name, {}).get("error", "not run"),
            }

    probe = HERE / "R17-H143_bagprobe.json"
    if probe.exists():
        # gate diagnostic: the prompt space searched before Baguettotron's
        # sub-chance control read was attributed to the model rather than the harness
        per_model["PleIAs/Baguettotron"]["gate_diagnostic"] = json.loads(probe.read_text())

    # ---------------- branch adjudication ---------------- #
    def pooled(n):
        return per_model.get(n, {}).get("pooled_auroc")

    in_budget = [
        n for n, r in per_model.items()
        if n not in (TEACHER, TEACHER_SUBSTITUTE, TEACHER_MEASURED) and n not in OVER_BUDGET
        and r.get("pooled_auroc") is not None and not r.get("skipped")
    ]
    best = max(in_budget, key=lambda n: pooled(n)) if in_budget else None
    best_auroc = pooled(best) if best else None
    teacher_auroc = pooled(TEACHER_MEASURED)

    fam_check = {}
    if best:
        fam = per_model[best].get("per_family_auroc", {})
        fam_check = {
            "families_used": top3,
            "families_excluding_relational": top3_no_rel,
            "per_family": {f: fam.get(f) for f in top3},
            "per_family_excluding_relational": {f: fam.get(f) for f in top3_no_rel},
            "passed": all((fam.get(f) or 0) >= FAMILY_BAR for f in top3),
            "passed_excluding_relational": all(
                (fam.get(f) or 0) >= FAMILY_BAR for f in top3_no_rel
            ),
        }

    sub_auroc = pooled(TEACHER_SUBSTITUTE)
    tiny_ok = best_auroc is not None and best_auroc >= TINY_BAR and fam_check.get("passed")
    if tiny_ok:
        branch = "TIER-VIABLE"
    elif teacher_auroc is None:
        # the registered teacher never ran, so no branch keyed on its bars can be
        # closed; the tiny half alone only rules TIER-VIABLE out
        branch = "GRAY-ZONE"
    elif teacher_auroc < TEACHER_FLOOR:
        branch = "ROUTE-KILLED"
    elif teacher_auroc >= TEACHER_BAR:
        branch = "DISTILL-LICENSED"
    else:
        branch = "GRAY-ZONE"

    branch_note = None
    if teacher_auroc is not None:
        branch_note = (
            f"TIER-VIABLE is ruled out on the tiny half alone (best in-budget {best_auroc}, "
            f"bar {TINY_BAR}). The registered teacher {TEACHER} has no cached weights; the "
            f"author's ordered completion measured the cached {TEACHER_MEASURED} through the "
            f"identical eval set, prompt and forced-answer discipline, and it reads "
            f"{teacher_auroc} pooled on "
            f"{per_model[TEACHER_MEASURED].get('control_auroc')} controls. Against the "
            f"registered teacher bars (distill {TEACHER_BAR}, route-kill floor "
            f"{TEACHER_FLOOR}) the branch is {branch}."
        )
    if teacher_auroc is None:
        would = "ROUTE-KILLED" if (sub_auroc or 0) < TEACHER_FLOOR else (
            "DISTILL-LICENSED" if (sub_auroc or 0) >= TEACHER_BAR else "GRAY-ZONE")
        branch_note = (
            f"TIER-VIABLE is ruled out on the tiny half alone (best in-budget "
            f"{best_auroc}, bar {TINY_BAR}). The registered teacher "
            f"{TEACHER} has no cached weights and was not measured, so no "
            f"teacher-keyed branch is closed. The substitute ceiling "
            f"{TEACHER_SUBSTITUTE} reads {sub_auroc}; were it the registered "
            f"teacher the branch would be {would}."
        )

    result = {
        "experiment": "R17-H143 Stage A - TINY-REASONER-RESIDUAL (measurement only)",
        "per_model": per_model,
        "branch_evaluation": {
            "best_in_budget_tiny": best,
            "best_tiny_pooled_auroc": best_auroc,
            "tiny_bar": TINY_BAR,
            "family_bar": FAMILY_BAR,
            "top3_families_check": fam_check,
            "teacher_auroc": teacher_auroc,
            "teacher_model": TEACHER_MEASURED,
            "teacher_params_M": PARAMS_M.get(TEACHER_MEASURED),
            "teacher_control_auroc": per_model.get(TEACHER_MEASURED, {}).get("control_auroc"),
            "teacher_per_family_auroc": per_model.get(TEACHER_MEASURED, {}).get(
                "per_family_auroc"),
            "teacher_parse_fail_rate": per_model.get(TEACHER_MEASURED, {}).get("parse_fail_rate"),
            "teacher_registered": TEACHER,
            "teacher_measured": teacher_auroc is not None,
            "teacher_substitute": TEACHER_SUBSTITUTE,
            "teacher_substitute_auroc": sub_auroc,
            "teacher_substitute_params_M": PARAMS_M.get(TEACHER_SUBSTITUTE),
            "teacher_bars": {"distill": TEACHER_BAR, "route_kill_floor": TEACHER_FLOOR},
            "branch": branch,
            "branch_note": branch_note,
            "qwen3_reference_only_auroc": pooled("Qwen/Qwen3-0.6B"),
        },
        "eval_set": {
            "n_pairs": int(ev.filter(~pl.col("control")).height),
            "n_controls": int(ev.filter(pl.col("control")).height),
            "seed": 1143,
            "source_file": SOURCE_FILE,
            "family_counts": fam_counts,
            "claim_form_counts": {
                k: v for k, v in
                ev.filter(~pl.col("control")).group_by("claim_form").len().iter_rows()
            },
        },
    }
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["branch_evaluation"], indent=2))
    for n, r in per_model.items():
        print(
            f"{n:45s} pooled={r.get('pooled_auroc')} ctrl={r.get('control_auroc')} "
            f"pf={r.get('parse_fail_rate')} skipped={r.get('skipped', False)}"
        )
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
