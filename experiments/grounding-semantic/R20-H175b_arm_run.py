"""R20-H175b QUESTION CONDITIONING (MEASUREMENT ONLY) - the training draw.

Registered in docs/experiments/semantic-grounding-experiments.md, blocks
"R20-H175b QUESTION CONDITIONING (measurement only)" (2026-08-16 ~23:30),
"R20-H175b STAGE 0 COMPLETE" (2026-08-17 ~00:10), "BASELINE LEGS BANKED"
(2026-08-17 ~05:45) and "QUEUE AMENDMENT Q1" (2026-08-17 ~06:05, DRAW 1 ONLY).

CLASSIFICATION: MEASUREMENT. There is no promotion route to the shipped
`ground()` / `ground_batch()` API; nothing under `src/groundrails/` is touched.
A pass produces a number and an author decision item, not a release.

    recipe    the R18-H150 flagship VERBATIM - evidence UNTRUNCATED presented as
              1,500/750 windows, MIL max-over-windows BCE, per-pair domain CE at
              DANN lambda 0.02 with the Ganin ramp, adapter FROZEN at its zero
              init, MAX_LEN 512, <=48 sets / <=96 pairs per batch, LR 1e-5
              OneCycleLR 10% linear warmup, clip 1.0, 1 epoch, H126 double
              seeding. NO EMA, NO window dropout
    mix       clean public 685,670 (R10-H108.public_train, untruncated)
              + H146 misbind        30,000  -> `quant_misbind`
              + H150 unit_swap       5,540  -> `quant_scale_unit`
              + H175b contrast lane 17,972  -> `qrel_contrast`
              = 739,182 rows, 15 DANN groups
    the ONE   an OPTIONAL question prefix on the CLAIM side of the cross-encoder
    change    input - `"<question[:256]> [SEP] <claim>"` where a question exists,
              the bare claim where it does not (`R20-H175b_qchannel.compose`).
              The evidence side, the windowing and the objective are untouched
    seed      1175 (draw 1). Further draws exist only if the mechanism gate passes
    ckpt      models/R20-H175b-arm-draw1
    results   R20-H175b_arm_draw1_result.json          (train + in-domain suite)
              R20-H175b_arm_draw1_windowed_result.json (blind arena read - and,
                                                        the arena carrying no
                                                        question field, ALSO the
                                                        empty-question hold)

THE MANDATORY LOADER ASSERTION (stage-0 disposition 4, registered before this
launch). Both rows of a `qrel_contrast` pair carry the SAME claim and the SAME
chunk; the label lives entirely in the question. Loaded into a mix that drops the
question the lane becomes label-contradictory duplicate rows - pure label noise
at ~2.4% of the mix - and the arm would measure nothing while looking healthy.
`build_mix` therefore hard-aborts unless a question is composed for EVERY lane
row, and proves POSITIVELY that the two legs of a pair differ as composed strings
(all 8,986 pairs) and as tokenized model inputs (a 400-pair sample). This is not
advisory and it runs before a card is touched.

THE CENSUS REBIND. The contrast lane moves the mix off the banked H150 geometry
(721,210 / 1.4821 / 0.1908), which `census_crosscheck` reads as a hard abort.
`R20-H175b_window_census.py` recomputes the combined census from the built mix,
asserts every component against a banked figure before writing, and this wrapper
repoints `census_crosscheck` at `R20-H175b_window_census.json`. The control is
REPOINTED, NEVER WEAKENED - a mix that drifts still aborts.

Everything else is banked code, imported not copied: the H150 mix assembly
(`R18-H150_arm_run.make_build_mix`, rebound to the 3-lane LANES tuple and then
wrapped with the question channel), the H160 draw wrapper
(`R19-H160_arm_run.rebind`), the cotangent split executor
(`R19-H160_split_exec`), and the G1 twin trainer/reader. Injection is R20-H174's.

Stages:
    train     train + the in-domain suite, through the split executor
    windowed  the blind windowed decomposed-min arena read
    census    CPU-only dry run - mix census, the rebound window-census
              cross-check, the draw's init and permutation fingerprints
    qproof    CPU-only - build the mix and write the intervention proof
              (question coverage, the loader assertion, the paired-leg
              tokenization evidence) to R20-H175b_intervention_proof.json

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 \
      uv run python experiments/grounding-semantic/R20-H175b_arm_run.py \
          --stage train --draw 1
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import polars as pl

HERE = pathlib.Path(__file__).parent

EXPECTED_CLEAN_ROWS = 685_670
EXPECTED_MIX_ROWS = 739_182

# sorted() order - `make_build_mix` compares tuple(sorted(set(tags))) to this
EXPECTED_GROUPS = (
    "halueval", "psiloqa", "qrel_contrast", "quant_misbind", "quant_scale_unit",
    "ragtruth_cn", "ragtruth_de", "ragtruth_en", "ragtruth_es", "ragtruth_fr",
    "ragtruth_hu", "ragtruth_it", "ragtruth_pl", "tabfact", "vitaminc",
)

# (file, DANN group, rows, pairs, {neg_family: count}) - the two flagship lanes
# at their banked scale plus the stage-0 contrast lane. Any drift aborts before a
# card is touched.
LANES = (
    ("R17-H146_lane.parquet", "quant_misbind", 30_000, 15_000,
     {"misbound_row": 21_000, "misbound_col": 9_000}),
    ("R18-H150_scaleunit_lane.parquet", "quant_scale_unit", 5_540, 2_770,
     {"unit_swap": 5_540}),
    ("R20-H175b_qlane.parquet", "qrel_contrast", 17_972, 8_986,
     {"qswap_same_passage": 17_972}),
)

# The re-banked combined census this arm's crosscheck reads.
WINDOW_CENSUS = HERE / "R20-H175b_window_census.json"
PROOF_OUT = HERE / "R20-H175b_intervention_proof.json"

DRAWS = {
    1: {"seed": 1175, "ckpt": "R20-H175b-arm-draw1",
        "train_out": "R20-H175b_arm_draw1_result.json",
        "read_out": "R20-H175b_arm_draw1_{mode}_result.json"},
}

# Every banked permutation fingerprint of this recipe absent from
# `R19-H160_arm_run.BANKED_PERM_FPS` (which covers the draws banked to R18-H156).
# The H174 launch widened it to H160 d3/d4 and H172 d5/d6 and recorded that the
# guard's coverage claim had been false since H156; this adds H156 d1, the two
# H174 draws now training, and the earlier-recipe draws on disk. Widening only
# strengthens the guard.
EXTRA_PERM_FPS = {
    "a867296772f8314a", "709afd02843c742e",   # R19-H160 d3 / d4
    "a8e708538a5decd8", "a4244751f7bb646b",   # R20-H172 d5 / d6
    "a75c4b59777d442d",                       # R18-H156 d1
    "ded543769d14f9e3", "a42b9d29e07c9db0",   # R20-H174 d1 / d2 (training now)
    "51dce43a8ae07065", "ad65e0d529fa257b",   # R14-H133 d1 / d2
    "1227e10c9daa2922", "90f8c77667a667ff",   # R14-H135 d1, R17-H145 d1
    "25bd6d194ff18cc6", "39e7dd9d4a12753f",   # R17-H146 d1, R19-H159 d1
}

# Populated by build_mix; written out by --stage qproof.
PROOF = {}


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


Q = _mod("qchan", "R20-H175b_qchannel.py")


def make_question_build_mix(banked_make, arm):
    """The banked H150 assembly, then the arm's one change: compose the optional
    question into the claim side. Every banked abort (lane composition, group
    map, row count, window census) has already fired by the time this runs."""
    base = banked_make(arm)

    def build_mix():
        claims, wsets, y, tags = base()

        # --- the clean mix's question channel, proved aligned row for row ---- #
        q_clean, replay, segments = Q.clean_questions()
        if len(q_clean) != EXPECTED_CLEAN_ROWS:
            raise SystemExit(
                f"QUESTION-CHANNEL ABORT: the replay produced {len(q_clean)} clean "
                f"rows, expected {EXPECTED_CLEAN_ROWS}")
        Q.assert_alignment(replay, claims[:EXPECTED_CLEAN_ROWS])
        print(f"question channel: clean-mix replay aligned row for row over "
              f"{EXPECTED_CLEAN_ROWS} rows ({len(segments)} source segments)",
              flush=True)
        questions = list(q_clean)
        del q_clean, replay

        # --- the lanes, in LANES order, same order the banked assembly used -- #
        at = EXPECTED_CLEAN_ROWS
        for fname, group, n_rows, _pairs, _fams in LANES:
            df = pl.read_parquet(HERE / fname)
            if df["claim"].to_list() != claims[at:at + n_rows]:
                raise SystemExit(
                    f"QUESTION-CHANNEL ABORT: lane {group} claims do not line up "
                    f"with the assembled mix at rows {at}..{at + n_rows}")
            if "question" in df.columns:
                questions += df["question"].to_list()
            else:
                questions += [""] * n_rows
            at += n_rows

        if len(questions) != len(claims) or len(claims) != EXPECTED_MIX_ROWS:
            raise SystemExit(
                f"QUESTION-CHANNEL ABORT: {len(questions)} questions for "
                f"{len(claims)} claims (mix expects {EXPECTED_MIX_ROWS})")

        # --- the intervention ------------------------------------------------ #
        composed = [Q.compose(q, c) for q, c in zip(questions, claims, strict=True)]

        # --- the MANDATORY loader assertion, and the positive verification --- #
        lane_report = Q.assert_lane_channel(questions, composed, claims, tags)
        tok = _tokenizer(arm)
        pair_report = Q.assert_pairs_differ(
            composed, tags, tokenizer=tok, chunk_of=lambda i: wsets[i][0])
        cov = Q.coverage(questions, tags)

        changed = sum(1 for a, b in zip(composed, claims, strict=True) if a != b)
        print(f"question channel: {cov['rows_with_question']}/{cov['mix_rows']} rows "
              f"carry a question ({cov['coverage']:.4f}); {changed} composed inputs "
              f"differ from the bare claim", flush=True)
        print(f"qlane loader assertion PASSED: {lane_report['lane_rows']}/"
              f"{Q.LANE_ROWS} rows with a question, all composed inputs changed",
              flush=True)
        print(f"qlane paired-leg check PASSED: {pair_report['pairs']} pairs differ "
              f"as strings, {pair_report.get('pairs_token_checked', 0)} sampled "
              f"pairs differ as TOKENIZED inputs (0 identical)", flush=True)

        PROOF.update({
            "experiment": "R20-H175b intervention proof - the question channel is "
                          "real on the built mix",
            "composition": f"'<question[:{Q.Q_MAX_CHARS}]>{Q.Q_SEP}<claim>' on the "
                           "claim side; bare claim where no question exists",
            "separator": Q.Q_SEP,
            "question_max_chars": Q.Q_MAX_CHARS,
            "mix_rows": len(claims),
            "composed_inputs_changed": changed,
            "coverage": cov,
            "per_source_segments": segments,
            "qlane_loader_assertion": lane_report,
            "qlane_paired_leg_check": pair_report,
        })
        return composed, wsets, y, tags

    return build_mix


def _tokenizer(arm):
    """The mix's tokenizer, loaded WITHOUT disturbing the RNG streams the H126
    double-seeding protocol relies on - model construction happens after this and
    must consume exactly the draws the flagship convention gives it."""
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    t_state = torch.get_rng_state()
    n_state = np.random.get_state()
    try:
        return AutoTokenizer.from_pretrained(arm.H108.STUDENT)
    finally:
        torch.set_rng_state(t_state)
        np.random.set_state(n_state)


def load_h150():
    """The banked H150 mix assembly with this arm's lanes, group map, row count
    and census target injected, then wrapped with the question channel. The
    closure resolves all four through the module's globals at call time, so
    nothing is copied."""
    if not WINDOW_CENSUS.exists():
        raise SystemExit(
            f"CENSUS REBIND ABORT: {WINDOW_CENSUS.name} is not on disk - run "
            "R20-H175b_window_census.py first (it asserts the flagship sub-mix "
            "against the banked H150 census and the contrast lane against its own "
            "manifest before writing)")
    h150 = _mod("h150arm", "R18-H150_arm_run.py")
    h150.LANES = LANES
    h150.EXPECTED_GROUPS = EXPECTED_GROUPS
    h150.EXPECTED_MIX_ROWS = EXPECTED_MIX_ROWS
    h150.WINDOW_CENSUS = WINDOW_CENSUS
    banked_make = h150.make_build_mix
    h150.make_build_mix = lambda arm: make_question_build_mix(banked_make, arm)
    return h150


def load_w160():
    """The H160 draw wrapper with the H175b draw installed, the permutation guard
    widened to every banked draw, and its H150 load routed to the injected mix
    assembly."""
    w160 = _mod("h160base", "R19-H160_arm_run.py")
    w160.DRAWS = dict(DRAWS)
    w160.BANKED_PERM_FPS |= EXTRA_PERM_FPS
    h150 = load_h150()
    orig_mod = w160._mod

    def routed(name, fname):
        if fname == "R18-H150_arm_run.py":
            return h150
        return orig_mod(name, fname)

    w160._mod = routed
    return w160


def relabel(draw):
    """The banked executor writes the H150 recipe's own description into its
    result file and its checkpoint fingerprint file. Correct the descriptive
    fields so the record names the arm that actually ran; every measured number
    is left untouched."""
    cfg = DRAWS[draw]
    p = HERE / cfg["train_out"]
    res = json.loads(p.read_text())
    res["arm"] = f"h175b_question_conditioning_draw{draw}"
    res["experiment"] = (
        f"R20-H175b question conditioning draw {draw} (MEASUREMENT ONLY) - the "
        "R18-H150 flagship recipe verbatim plus an optional question prefix on "
        "the claim side and the stage-0 qrel_contrast lane")
    res["mix"] = ("clean public mix (R10-H108.public_train) + R17-H146 misbind 30,000 "
                  "+ R18-H150 unit_swap 5,540 + R20-H175b qrel_contrast 17,972")
    res["clean_rows"] = EXPECTED_CLEAN_ROWS
    res["lane_rows"] = {g: n for _f, g, n, _p, _fam in LANES}
    res["lane_groups"] = [g for _f, g, _n, _p, _fam in LANES]
    res["window_census_source"] = WINDOW_CENSUS.name
    res["question_channel"] = {
        "composition": f"'<question[:{Q.Q_MAX_CHARS}]>{Q.Q_SEP}<claim>'",
        "coverage": PROOF.get("coverage", {}).get("coverage"),
        "rows_with_question": PROOF.get("coverage", {}).get("rows_with_question"),
        "proof": PROOF_OUT.name,
    }
    res["in_domain_note"] = (
        "the in-domain suite is read with NO question - it is therefore the "
        "empty-question robustness read, not a question-conditioned one")
    res["classification"] = ("MEASUREMENT ONLY - no promotion route to the shipped "
                             "ground()/ground_batch() API")
    res["bars_note"] = ("the bars/control blocks are the banked G1 twin's; H175b's "
                        "registered bars, mechanism gate and guards are adjudicated "
                        "by the coordinator")
    p.write_text(json.dumps(res, indent=2))
    print(f"result relabelled -> {p}", flush=True)

    fpp = HERE.parent.parent / "models" / cfg["ckpt"] / "init_fingerprint.json"
    if fpp.exists():
        fpj = json.loads(fpp.read_text())
        fpj["arm"] = f"h175b_question_conditioning_draw{draw}"
        fpj["recipe"] = ("R20-H175b question conditioning - R18-H150 flagship "
                         "verbatim (clean 685,670 + misbind 30,000 + unit_swap "
                         "5,540) + qrel_contrast 17,972, with an optional question "
                         "prefix on the claim side; MIL max-BCE; no EMA, no window "
                         "dropout")
        fpp.write_text(json.dumps(fpj, indent=2))
        print(f"checkpoint fingerprint relabelled -> {fpp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=("train", "windowed", "census", "qproof"))
    ap.add_argument("--draw", type=int, required=True, choices=tuple(DRAWS))
    ap.add_argument("--max-steps", type=int, default=0,
                    help="smoke mode for --stage train: stop after N steps")
    args = ap.parse_args()

    w160 = load_w160()
    cfg = w160.DRAWS[args.draw]

    if args.stage == "train":
        # The executor re-loads "R19-H160_arm_run.py" through its own `_mod`;
        # route that load to the injected wrapper (the R20-H172 pattern).
        split = _mod("h160split", "R19-H160_split_exec.py")
        orig_mod = split._mod

        def routed(name, fname):
            if fname == "R19-H160_arm_run.py":
                return w160
            return orig_mod(name, fname)

        split._mod = routed
        split.train(args.draw, max_steps=args.max_steps)
        if not args.max_steps:
            relabel(args.draw)
        return

    if args.stage == "windowed":
        reads = _mod("g1reads", "R16-H142_G1_reads.py")
        w160.rebind(reads.ARM, args.draw)
        reads.out_path = lambda run, mode: HERE / cfg["read_out"].format(mode=mode)
        sys.argv = ["reads", "--run", "twin", "--mode", "windowed"]
        reads.main()
        return

    if args.stage == "qproof":
        arm = w160.rebind(_mod("g1arm", "R16-H142_G1_arm.py"), args.draw)
        arm.build_mix()
        PROOF["draw"] = args.draw
        PROOF["seed"] = cfg["seed"]
        PROOF["note"] = ("Numbers recorded, not adjudicated - the coordinator "
                         "adjudicates.")
        PROOF_OUT.write_text(json.dumps(PROOF, indent=2))
        print(f"intervention proof -> {PROOF_OUT}", flush=True)
        print("=== R20-H175b INTERVENTION PROOF WRITTEN ===", flush=True)
        return

    # census - CPU dry run through the banked arm module
    arm = w160.rebind(_mod("g1arm", "R16-H142_G1_arm.py"), args.draw)
    sys.argv = ["arm", "--run", "twin", "--census-only"]
    arm.main()


if __name__ == "__main__":
    main()
