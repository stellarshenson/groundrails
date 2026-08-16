"""R19 CORRECTIONS CENSUS - the artifact the corrections block should have had.

The first correction pass asserted two counts in prose with no script and no JSON
behind them. A confirming adversarial round then found one of them wrong. That is
exactly the failure the campaign's own artifact discipline exists to prevent, and
it was self-inflicted: every arm in this log banks its numbers, and the block
CORRECTING those arms did not.

This banks the three counts the corrections depend on, so they can be re-derived
from the tree rather than trusted.

  A. nonEN-minus-EN census across every banked checkpoint - the evidence for
     withdrawing R19-H168's "the adversary won outright" mechanism
  B. per-subset win/loss against the incumbent under BOTH conventions - the
     consequence of the R19-H171 margin correction that the first pass missed
  C. the H165 subset-leg null spreads, both the paired and the across-seed
     reading, so the bar critique rests on a stated comparison rather than an
     implied one

Run: uv run python experiments/grounding-semantic/R19_corrections_census.py
"""

import glob
import json
import os
import pathlib

HERE = pathlib.Path(__file__).parent
TIE_BAND = 0.005          # |delta| within this counts as a tie, stated not implied


def census_nonen_en():
    """A. Every banked checkpoint carrying both RAGTruth gates, deduped by value pair."""
    seen, rows = set(), []
    for f in sorted(glob.glob(str(HERE / "*_result.json"))):
        try:
            d = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue

        def walk(o):
            if isinstance(o, dict):
                en, ne = o.get("ragtruth_en"), o.get("ragtruth_nonen")
                if isinstance(en, dict) and isinstance(ne, dict):
                    a, b = en.get("auc"), ne.get("auc")
                    if a is not None and b is not None:
                        yield a, b
                for v in o.values():
                    yield from walk(v)

        for en, ne in walk(d):
            if (en, ne) in seen:
                continue
            seen.add((en, ne))
            rows.append({"file": os.path.basename(f), "en": en, "nonen": ne,
                         "delta": round(ne - en, 4)})
    rows.sort(key=lambda r: r["delta"])
    pos = sum(r["delta"] > 0 for r in rows)
    neg = sum(r["delta"] < 0 for r in rows)
    return {"n_checkpoints": len(rows), "n_nonen_above_en": pos,
            "n_reversals": neg,
            "largest_reversal": rows[0] if rows else None,
            "largest_positive": rows[-1] if rows else None,
            "reversals": [r for r in rows if r["delta"] < 0],
            "reading": ("nonEN above EN is the DOMINANT direction, not universal - "
                        "reversals exist and the largest is larger in magnitude than "
                        "several of the positive deltas"),
            "rows": rows}


def per_subset_vs_incumbent():
    """B. Win/loss under the harness convention and under the vendor's own."""
    nat = json.load(open(HERE / "R19-H171_incumbent_native.json"))["per_subset"]
    d1 = json.load(open(HERE / "R18-H150_arm_draw1_windowed_result.json"))["per_subset"]
    d2 = json.load(open(HERE / "R18-H150_arm_draw2_windowed_result.json"))["per_subset"]
    out = {}
    for conv in ("harness_auc", "native_auc"):
        rows, w, l, t = {}, 0, 0, 0
        for s in sorted(nat):
            ours = (d1[s]["auc"] + d2[s]["auc"]) / 2
            inc = nat[s][conv]
            d = ours - inc
            v = "win" if d > TIE_BAND else ("loss" if d < -TIE_BAND else "tie")
            w += v == "win"
            l += v == "loss"
            t += v == "tie"
            rows[s] = {"ours_2draw_mean": round(ours, 4), "incumbent": inc,
                       "delta": round(d, 4), "verdict": v}
        out[conv] = {"wins": w, "losses": l, "ties": t, "tie_band": TIE_BAND,
                     "losing_subsets": sorted(s for s, r in rows.items()
                                              if r["verdict"] == "loss"),
                     "per_subset": rows}
    return out


def h165_null_spreads():
    """C. Both readings of the H165 subset leg's null - stated side by side."""
    d1 = json.load(open(HERE / "R18-H150_arm_draw1_windowed_result.json"))["per_subset"]
    d2 = json.load(open(HERE / "R18-H150_arm_draw2_windowed_result.json"))["per_subset"]
    across = {s: round(d2[s]["auc"] - d1[s]["auc"], 4) for s in sorted(d1)}
    l0 = json.load(open(HERE / "R19-H165_ladder_L0_R18-H150-arm-draw1.json"))
    return {
        "paired_same_checkpoint_same_presentation": {
            "value": l0["positive_control"]["abs_delta"],
            "note": ("the null that MATCHES the bar's own pairing - the bar compares "
                     "one checkpoint under two presentations, so its null is that "
                     "checkpoint re-read under one presentation. Deterministic."),
        },
        "across_seed_same_recipe": {
            "per_subset": across,
            "n_exceeding_0.01": sum(1 for v in across.values() if v < -0.01),
            "note": ("a DIFFERENT null - two seeds of the same recipe. Informative "
                     "about how much a subset moves between draws, but NOT the null "
                     "of a paired within-checkpoint bar."),
        },
        "reading": ("The first correction pass used the across-seed spread to claim "
                    "the subset leg 'fails on doing nothing'. That substitutes an "
                    "unpaired null for a paired bar and is withdrawn. What survives: "
                    "the mean leg and the subset leg are not independent evidence, and "
                    "a 0.01 leg is tight relative to how much these subsets move "
                    "between draws."),
    }


def main():
    res = {"what": "artifact backing the R19 corrections wave",
           "why": ("the first corrections block asserted counts in prose with no "
                   "script; a confirming review found one wrong. Banked so the "
                   "corrections meet the same standard as the arms they correct."),
           "A_nonen_en_census": census_nonen_en(),
           "B_per_subset_vs_incumbent": per_subset_vs_incumbent(),
           "C_h165_null_spreads": h165_null_spreads()}
    out = HERE / "R19_corrections_census.json"
    out.write_text(json.dumps(res, indent=1))

    a = res["A_nonen_en_census"]
    print(f"A. nonEN>EN in {a['n_nonen_above_en']}/{a['n_checkpoints']} checkpoints, "
          f"{a['n_reversals']} reversals, largest reversal "
          f"{a['largest_reversal']['delta']} ({a['largest_reversal']['file']})", flush=True)
    for conv, v in res["B_per_subset_vs_incumbent"].items():
        print(f"B. {conv:12} {v['wins']}W/{v['losses']}L/{v['ties']}T  "
              f"losses: {v['losing_subsets']}", flush=True)
    c = res["C_h165_null_spreads"]
    print(f"C. paired null {c['paired_same_checkpoint_same_presentation']['value']}  |  "
          f"across-seed subsets moving >0.01: "
          f"{c['across_seed_same_recipe']['n_exceeding_0.01']}/10", flush=True)
    print(f"  -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
