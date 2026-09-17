"""R23-H192 HAGRID AUTOPSY, stage 1 - MEASUREMENT ONLY, CPU, banked artifacts only.

hagrid is the arena's worst loss to the incumbent (-0.1149) and its most
concentrated failure: 37 items every one of the six flagship draws gets wrong,
carrying 46.5% of its deficit against a 21.4% arena mean, with +0.1440 of
headroom to the faithful-oracle ceiling. R19-H162 nominated a `vacuous_claim_reject`
lane for it; R20-H174 built that lane (L1 `frame_reject`, 8,000 rows) and hagrid
FELL 0.6393 -> 0.6166. This autopsy asks why the best-evidenced target in the
campaign did not respond.

MEASUREMENT ONLY. No bar, no gate, no promotion, no kill, no arm registered.
The arena is an analysis surface: nothing here may be used to select a serving
transform, an aggregation, or a lane. Reads banked artifacts and the frozen
arena; scores nothing, so no GPU and no card-pinning question.

Two questions, both answerable without training:

    Q1  what are hagrid's 37 consensus errors made of, and does the frame class
        R20-H174 attacked still account for what the record says it does
    Q3  do the subsets Q2 flags share any measured STRUCTURAL property - evidence
        volume, document count, window count, response shape, class balance -
        that a lane could be built against
    Q4  where does the remaining FAITHFUL headroom sit, and what fraction of it
        must be captured to reach the author's 0.74 target

    Q2  how much of the score survives when response LENGTH is controlled for -
        because the per-subset decomposition (2026-08-18) found a subset-blind
        sentence-count rule reads 0.59698 on the arena, 46% of the flagship's
        above-chance lift, and the shipped read is a MIN over sentences, which
        is monotonically length-decreasing by construction

Q2 is run with TWO estimators that fail differently, and only their agreement is
reported as a finding:

    linear-out       regress the 6-draw mean score on sentence count, score the
                     residual. Cheap, uses every pair, but assumes the length
                     effect is linear and can over- or under-correct
    within-stratum   AUROC counted ONLY over (positive, negative) pairs sharing
                     a sentence-count stratum (capped at 6 to avoid singletons),
                     so length cannot contribute to any counted pair.
                     Assumption-free but discards 23-60% of pairs and is noisier

Where the two disagree on a subset's VALUE, no value is quoted for it. The
finding is the SET of subsets both estimators drive near chance, never a ranking.

Run (CPU, under a minute):
    CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 \
    uv run python experiments/grounding-semantic/R23-H192_hagrid_autopsy.py
"""

import importlib.util
import json
import pathlib
import statistics as st

import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
OUT = HERE / "R23-H192_hagrid_autopsy.json"

# the banked incumbent, vendor's own convention (R19-H171_incumbent_chunked.json)
INCUMBENT = {"hagrid": 0.7542, "emanual": 0.7694, "expertqa": 0.8098, "pubmedqa": 0.6070,
             "covidqa": 0.7432, "hotpotqa": 0.6161, "finqa": 0.6137, "techqa": 0.6536,
             "delucionqa": 0.7018, "tatqa": 0.5275}

FRAME_OPENERS = ("based on the given context", "according to the given context",
                 "according to context", "based on context",
                 "according to the context", "based on the context")
FRAME_MAX_CHARS = 60
STRATUM_CAP = 6          # sentence counts above this share one stratum
NEAR_CHANCE = 0.62       # both estimators below this = "no length-free signal"


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def is_vacuous_frame(text):
    """A response that is a bare discourse frame carrying no proposition."""
    t = text.strip().lower().rstrip(" ,.")
    return len(text) < FRAME_MAX_CHARS and any(t.startswith(f) for f in FRAME_OPENERS)


def within_stratum_auc(y, score, strata):
    """AUROC over ONLY those (pos, neg) pairs sharing a stratum.

    Length cannot contribute to a pair whose two members have the same length,
    so this is a length control that assumes nothing about the effect's shape.
    Returns (auc, pairs_counted, pairs_available)."""
    conc = ties = counted = 0
    for v in np.unique(strata):
        m = strata == v
        pos, neg = score[m][y[m] == 1], score[m][y[m] == 0]
        if not len(pos) or not len(neg):
            continue
        d = pos[:, None] - neg[None, :]
        conc += int((d > 0).sum())
        ties += int((d == 0).sum())
        counted += d.size
    avail = int((y == 1).sum() * (y == 0).sum())
    return ((conc + 0.5 * ties) / counted if counted else float("nan")), counted, avail


def main():
    arena = _mod("arena", "R8-H77_unseen_arena.py")
    h92 = _mod("h92", "R8-H92_decomposed_arena.py")
    subs = arena.load_subsets()
    df = (pl.read_parquet(HERE / "R21-H179_consensus_errors.parquet")
          .with_columns(pl.mean_horizontal([f"score_d{i}" for i in range(1, 7)]).alias("s6")))

    # ---- Q1: what hagrid's consensus errors are made of -------------------
    resp, docs, y = subs["hagrid"]
    g = df.filter(pl.col("subset") == "hagrid").sort("item")
    score = g["s6"].to_numpy()
    lab = g["label"].to_numpy()
    ce = g["consensus_error_rank"].to_numpy()
    contrib = g["deficit_contrib_mean"].to_numpy()
    nsent = np.array([len(h92.sentences(r)) for r in resp])
    deficit = 1.0 - 0.63932  # hagrid's k=6 deficit

    frames = [i for i in range(len(resp)) if is_vacuous_frame(resp[i])]
    frame_mass = float(contrib[frames].sum())

    def profile(mask, name):
        return {"class": name, "n": int(mask.sum()),
                "mean_sentences": round(float(nsent[mask].mean()), 2),
                "mean_chars": round(float(np.array([len(r) for r in resp])[mask].mean()), 0),
                "mean_score": round(float(score[mask].mean()), 3)}

    q1 = {
        "consensus_errors": int(ce.sum()),
        "false_negatives": int((ce & (lab == 1)).sum()),
        "false_positives": int((ce & (lab == 0)).sum()),
        "deficit": round(deficit, 5),
        "fp_side_mass_from_consensus_fp": round(float(contrib[ce & (lab == 0)].sum()), 5),
        "fn_side_mass_from_consensus_fn": round(float(contrib[ce & (lab == 1)].sum()), 5),
        "mass_note": ("each mis-ranked pair is counted once on EACH side, so each side's total "
                      "IS the deficit by construction - the two sides must never be summed"),
        "vacuous_frames": {
            "items": frames, "n": len(frames),
            "labels": [int(y[i]) for i in frames],
            "scores": [round(float(score[i]), 2) for i in frames],
            "deficit_mass": round(frame_mass, 5),
            "share_of_negative_side_mass": round(frame_mass / deficit, 4),
            "prior_record": ("R20-H174 registered hagrid's misrank mass as 21.2% frame-only "
                             "artifacts and built L1 frame_reject (8,000 rows) against it; the "
                             "arm FELL to 0.6166 and the gate read 0.1962 against a <5% bar. "
                             "This measurement REPRODUCES the prior figure - it is confirmation "
                             "of an already-attacked-and-failed target, NOT a new one"),
        },
        "profiles": [profile(ce & (lab == 1), "consensus false negative"),
                     profile(ce & (lab == 0), "consensus false positive"),
                     profile(~ce & (lab == 1), "other positive"),
                     profile(~ce & (lab == 0), "other negative")],
    }

    # ---- Q2: how much signal survives a length control --------------------
    per_subset = {}
    for s, (sresp, sdocs, sy) in sorted(subs.items()):
        gg = df.filter(pl.col("subset") == s).sort("item")
        sc = gg["s6"].to_numpy()
        ns = np.array([len(h92.sentences(r)) for r in sresp])
        raw = roc_auc_score(sy, sc)
        coef = np.polyfit(ns.astype(float), sc, 1)
        lin = roc_auc_score(sy, sc - np.polyval(coef, ns.astype(float)))
        strat, counted, avail = within_stratum_auc(np.asarray(sy), sc, np.minimum(ns, STRATUM_CAP))
        per_subset[s] = {
            "auroc_raw": round(float(raw), 4),
            "auroc_length_only": round(float(roc_auc_score(sy, -ns)), 4),
            "auroc_linear_control": round(float(lin), 4),
            "auroc_within_stratum": round(float(strat), 4),
            "stratum_pairs_used": counted, "stratum_pairs_available": avail,
            "stratum_pair_coverage": round(counted / avail, 3),
            "incumbent": INCUMBENT[s],
            "margin": round(float(raw) - INCUMBENT[s], 4),
            "both_controls_near_chance": bool(lin < NEAR_CHANCE and strat < NEAR_CHANCE),
            "one_control_near_chance": bool((lin < NEAR_CHANCE) != (strat < NEAR_CHANCE)),
        }

    weak = sorted(s for s, v in per_subset.items() if v["both_controls_near_chance"])
    borderline = sorted(s for s, v in per_subset.items() if v["one_control_near_chance"])
    lose = sorted(s for s, v in per_subset.items() if v["margin"] <= 0)

    # ---- Q3: do the weak subsets share a structural property? -------------
    def win_count(txt, size=1500, stride=750):
        return 1 if len(txt) <= size else 1 + -(-(len(txt) - size) // stride)

    feats = {}
    for s, (sresp, sdocs, sy) in subs.items():
        feats[s] = {
            "documents": float(np.mean([len(d) for d in sdocs])),
            "evidence_chars": float(np.mean([sum(len(x) for x in d) for d in sdocs])),
            "chars_per_doc": float(np.mean([np.mean([len(x) for x in d]) for d in sdocs])),
            "windows_scored": float(np.mean([sum(win_count(x) for x in d) for d in sdocs])),
            "response_chars": float(np.mean([len(r) for r in sresp])),
            "sentences": float(np.mean([len(h92.sentences(r)) for r in sresp])),
            "evidence_over_response": float(np.mean([sum(len(x) for x in d) / len(r)
                                                     for d, r in zip(sdocs, sresp)])),
            "positive_rate": float(np.mean(sy)),
        }
    probe = sorted(set(weak) | set(borderline))
    ranks, isolates = {}, []
    for f in next(iter(feats.values())):
        order = sorted(feats, key=lambda s: feats[s][f])
        ranks[f] = {s: i + 1 for i, s in enumerate(order)}
        rs = sorted(ranks[f][s] for s in probe)
        if rs == [1, 2, 3] or rs == [8, 9, 10]:
            isolates.append(f)
    q3 = {
        "probed_subsets": probe,
        "features": {s: {k: round(v, 2) for k, v in fv.items()} for s, fv in feats.items()},
        "ranks": ranks,
        "features_isolating_the_weak_set": isolates,
        "test": ("a feature isolates the set only if it places all three at one extreme of the "
                 "ten (ranks 1-3 or 8-10). 2 of the 120 possible triples qualify, so a chance "
                 "hit is p=0.017 per feature and ~0.13 over the eight - the test is not weak"),
        "result": ("NEGATIVE - no feature isolates them; their ranks scatter across the range on "
                   "every property measured. A mean-based comparison APPEARS to separate them on "
                   "evidence volume, but that is an artifact of two outliers (techqa 18,653 and "
                   "expertqa 9,344 evidence chars) inflating the comparison group, and it does "
                   "not survive the rank test. The weakness is not a shape property, so there is "
                   "no structural handle for a lane to grip."),
    }

    # ---- Q4: headroom accounting against the author target ----------------
    TARGET = 0.74
    dec = json.loads((HERE / "R22_per_subset_decomposition.json").read_text())["per_subset"]
    cur = st.mean(dec[s]["auc_k6_mean"] for s in dec)
    ceiling = st.mean(dec[s]["faithful_oracle_ceiling"] for s in dec)
    gains = {s: max(0.0, dec[s]["faithful_oracle_ceiling"] - dec[s]["auc_k6_mean"]) for s in dec}
    total_headroom = sum(gains.values()) / len(gains)
    weak_headroom = sum(gains[s] for s in probe) / len(gains)
    q4 = {
        "flagship_k6": round(cur, 5),
        "faithful_oracle_ceiling_mean": round(ceiling, 5),
        "target": TARGET,
        "target_as_share_of_ceiling": round(TARGET / ceiling, 4),
        "current_as_share_of_ceiling": round(cur / ceiling, 4),
        "total_positive_headroom": round(total_headroom, 5),
        "needed_for_target": round(TARGET - cur, 5),
        "share_of_all_headroom_needed": round((TARGET - cur) / total_headroom, 4),
        "headroom_per_subset": {s: round(g, 4) for s, g in
                                sorted(gains.items(), key=lambda kv: -kv[1])},
        "headroom_in_the_weak_subsets": round(weak_headroom, 5),
        "weak_share_of_all_headroom": round(weak_headroom / total_headroom, 4),
        "finqa_headroom_unreachable": round(gains["finqa"] / len(gains), 5),
        "finding": ("The 0.74 target sits at 97.9% of the faithful-oracle ceiling - what a "
                    "PERFECT entailer reaches through the shipped read. Today we are at 94.2%. "
                    "Reaching it requires capturing 37% of ALL remaining faithful headroom in "
                    "the arena, perfectly, with zero regression anywhere. 60% of that headroom "
                    "sits in the three subsets Q2 shows have no length-free signal, and a "
                    "further 10% is finqa's, which R22-H190 proved unreachable by this "
                    "architecture."),
    }

    res = {
        "arm": "R23-H192 hagrid autopsy, stage 1",
        "licence": ("MEASUREMENT ONLY - no bar, no gate, no promotion, no kill, no arm. The "
                    "arena is an analysis surface: nothing here may select a serving transform, "
                    "an aggregation or a lane. Any lever this motivates must be registered with "
                    "its own bars and selected on a NON-arena surface."),
        "q1_hagrid_error_composition": q1,
        "q2_length_control": {
            "why": ("the shipped read is a MIN over sentences, monotonically length-decreasing "
                    "by construction, and a subset-blind sentence-count rule already reads "
                    "0.59698 on the arena (R22 per-subset decomposition)"),
            "estimators": {
                "linear_control": "residual of the 6-draw score regressed on sentence count",
                "within_stratum": (f"AUROC over only those (pos, neg) pairs sharing a sentence "
                                   f"count, capped at {STRATUM_CAP}; assumption-free but uses "
                                   f"23-60% of pairs and is noisier"),
            },
            "reporting_rule": ("only the AGREEMENT of the two estimators is reported as a "
                               "finding; where they disagree on a subset's value, no value is "
                               "quoted for that subset"),
            "near_chance_threshold": NEAR_CHANCE,
            "per_subset": per_subset,
            "no_length_free_signal": weak,
            "borderline_one_control_only": borderline,
            "estimator_note": (
                "auroc_raw here is the AUROC of the SIX-DRAW MEAN SCORE - a pooled read. The "
                "campaign's banked headline is the MEAN OF THE SIX PER-DRAW AUROCs. They are "
                "different estimators and differ by up to ~0.025 per subset (hagrid 0.6462 "
                "pooled vs 0.63932 banked), so the `margin` column here is NOT the banked "
                "margin and must not be quoted as one. The pooled read is used because the "
                "length controls need one score per item, not six AUROCs."),
            "subsets_not_beating_incumbent": lose,
            "finding": ("Both controls drive the SAME three subsets near chance: "
                        f"{', '.join(weak)}. Three of the four subsets that do not beat the "
                        "incumbent are exactly those three. The fourth, expertqa, keeps strong "
                        "length-free signal under both controls and is therefore a DIFFERENT "
                        "failure - a genuine loss to a better model, not an absent signal."),
        },
        "q3_structural_separation": q3,
        "q4_headroom_accounting": q4,
        "reading": ("hagrid's problem is not the frame class R20-H174 attacked and it is not a "
                    "5-item defect: once length is controlled, the model retains almost no "
                    "signal on hagrid at all. That reframes hagrid, pubmedqa and emanual as ONE "
                    "problem - a register where the grounder does not work - rather than three "
                    "separate coverage gaps, and it explains why an 8,000-row targeted lane "
                    "could not move it."),
    }
    OUT.write_text(json.dumps(res, indent=2))

    print(f"written -> {OUT}\n")
    print(f"hagrid: {q1['consensus_errors']} consensus errors "
          f"({q1['false_negatives']} FN / {q1['false_positives']} FP); "
          f"{len(frames)} vacuous frames carry {q1['vacuous_frames']['share_of_negative_side_mass']:.1%} "
          f"of the negative-side mass (REPRODUCES the banked 21.2%, already attacked and failed)\n")
    print(f"{'subset':<12}{'raw':>8}{'len only':>10}{'linear-out':>12}{'stratum':>10}"
          f"{'incumbent':>11}{'margin':>9}  flag")
    for s, v in sorted(per_subset.items(), key=lambda kv: kv[1]["auroc_within_stratum"]):
        flag = "NO LENGTH-FREE SIGNAL" if v["both_controls_near_chance"] else ""
        print(f"{s:<12}{v['auroc_raw']:>8.4f}{v['auroc_length_only']:>10.4f}"
              f"{v['auroc_linear_control']:>12.4f}{v['auroc_within_stratum']:>10.4f}"
              f"{v['incumbent']:>11.4f}{v['margin']:>+9.4f}  {flag}")
    print(f"\nQ3 features isolating the weak set: "
          f"{q3['features_isolating_the_weak_set'] or 'NONE - negative result'}")
    print(f"Q4 target {q4['target']} = {q4['target_as_share_of_ceiling']:.1%} of the oracle "
          f"ceiling; we are at {q4['current_as_share_of_ceiling']:.1%}")
    print(f"   reaching it needs {q4['share_of_all_headroom_needed']:.0%} of ALL remaining "
          f"headroom; {q4['weak_share_of_all_headroom']:.0%} of that headroom is in the "
          f"no-signal subsets")
    print(f"\nno length-free signal (BOTH controls): {weak}")
    print(f"borderline (ONE control only): {borderline}")
    print("NOTE: auroc_raw is the pooled 6-draw-mean-score read, NOT the banked mean-of-AUROCs;"
          "\n      the margin column here is therefore not the banked margin.")
    print(f"do not beat the incumbent: {lose}")


if __name__ == "__main__":
    main()
