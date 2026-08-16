"""R19-H169 EUROBERT WITHOUT THE ADVERSARY - separating the two readings of H168.

R19-H168 applied the flagship recipe to EuroBERT-210m and the model landed at
CHANCE in-domain (`gold_full` 0.5070 against the flagship's 0.8659) and 0.54498
blind. Training did not diverge - task loss fell 0.8787 -> 0.5721 - so the
failure is not an exploding run. The adversary is where the damage is visible:
domain accuracy ran 0.118 -> 0.346 by step 600 as the Ganin ramp engaged, then
COLLAPSED to 0.001 by step 14,000 with domain loss climbing to 7.87. With 14
groups the chance floor is 0.071, so 0.001 is far BELOW chance - gradient
reversal drove the trunk into a state actively scrubbed of domain information.
The corroborating fingerprint is multilingual: `ragtruth_nonen` 0.7712 spread
over only 0.0216 across seven languages, while `ragtruth_en` came in LOWER at
0.6194. A model whose English loses its home advantage is a model the adversary
won outright.

Two readings survive H168 and it cannot separate them:

  (a) DANN at LAMBDA_MAX 0.02 is too strong for a Llama-architecture encoder
      (RMSNorm, no biases, SwiGLU), where the reversed gradient's scale relative
      to the task gradient differs from ModernBERT's
  (b) LR 1e-5 on this OneCycle schedule simply does not suit the architecture,
      and the domain collapse is a symptom of general representation collapse
      rather than its cause

THIS ARM IS THE SEPARATOR. One variable against H168: LAMBDA_MAX 0.02 -> 0.0.
The domain head still exists and still trains on its own cross-entropy; only the
reversed gradient reaching the trunk is switched off. Everything else - mix,
seed 1150, schedule, LR, objective, window presentation, MAX_LEN, adapter off -
is H168's, which is in turn the flagship's, taken from `R18-H150_arm_run.rebind`.

VERDICT RULE, fixed before the read
-----------------------------------
  RECIPE       `gold_full` >= 0.75 - the adversary was the destroyer, EuroBERT is
               a viable trunk, and H168's kill says nothing about the encoder.
               The campaign also learns that lambda 0.02 is not architecture-portable
  ARCHITECTURE `gold_full` <= 0.60 - still at or near chance without any adversary,
               so the mismatch is deeper than DANN and reading (b) stands
  PARTIAL      in between - the adversary contributed but is not the whole story

WHY THIS IS WORTH 3.25 GPU-h
----------------------------
The author ordered a EuroBERT-versus-mmBERT comparison. H168 did not deliver one:
it delivered a recipe failure, and a model at chance in-domain cannot be compared
with anything. This run is what makes the ordered comparison interpretable. It
also gives the campaign its FIRST controlled evidence about the DANN adversary,
which the canonical log has never ablated on any trunk.

NOTE: this is a DIAGNOSTIC. It has no promotion path and no arena bar. A no-DANN
checkpoint could not ship regardless of its number - the non-English holds exist
because the adversary is believed to earn them - so no blind arena read is spent
here unless the coordinator registers one separately.

Run:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu> \
      uv run python experiments/grounding-semantic/R19-H169_eurobert_nodann.py
"""

import importlib.util
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent

CKPT = "R19-H169-eurobert-nodann"
TRAIN_OUT = "R19-H169_eurobert_nodann_result.json"
BARS = {"recipe_at_or_above": 0.75, "architecture_at_or_below": 0.60}
H168_GOLDFULL = 0.5070
FLAGSHIP_GOLDFULL = 0.8659


def _mod(name, fname):
    spec = importlib.util.spec_from_file_location(name, HERE / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


H168 = _mod("h168run", "R19-H168_arm_run.py")


def main():
    out = HERE / TRAIN_OUT
    if out.exists() and out.stat().st_size > 0:
        print(f"SKIP (on disk: {out.name})", flush=True)
        print("=== H169 COMPLETE ===", flush=True)
        return

    H168.verify_shim()

    # H168's rebind = flagship recipe + the EuroBERT trunk swap, from their own files
    arm = H168.rebind(_mod("g1arm", "R16-H142_G1_arm.py"), "h169")

    # --- the ONLY variable against H168 --------------------------------------
    was = arm.LAMBDA_MAX
    arm.LAMBDA_MAX = 0.0
    print(f"[h169] LAMBDA_MAX {was} -> {arm.LAMBDA_MAX} "
          f"(domain head still trains; no reversed gradient reaches the trunk)",
          flush=True)
    if arm.LAMBDA_MAX != 0.0:
        raise SystemExit("H169 ABORT: the adversary was not disabled")

    arm.RUNS["twin"]["ckpt"] = CKPT
    arm.RUNS["twin"]["out"] = TRAIN_OUT

    sys.argv = ["arm", "--run", "twin"]
    arm.main()

    # --- adjudicate against the pre-registered rule ---------------------------
    res = json.loads(out.read_text())
    gf = res["gold_full"]["auc"]
    if gf >= BARS["recipe_at_or_above"]:
        verdict = "RECIPE"
    elif gf <= BARS["architecture_at_or_below"]:
        verdict = "ARCHITECTURE"
    else:
        verdict = "PARTIAL"
    res["arm"] = "h169_eurobert_nodann"
    res["experiment"] = ("R19-H169 EuroBERT-210m with the DANN adversary disabled - "
                         "the separator for R19-H168's two readings")
    res["single_variable"] = "LAMBDA_MAX 0.02 -> 0.0 against R19-H168; nothing else"
    res["diagnostic_only"] = ("no promotion path, no arena bar, no blind read - a "
                              "no-adversary checkpoint could not ship regardless of "
                              "its number")
    res["h169_verdict"] = {
        "verdict": verdict, "gold_full": gf, "bars": BARS,
        "h168_gold_full": H168_GOLDFULL, "flagship_gold_full": FLAGSHIP_GOLDFULL,
        "delta_vs_h168": round(gf - H168_GOLDFULL, 5),
        "rule": ("RECIPE if gold_full >= 0.75 (the adversary was the destroyer); "
                 "ARCHITECTURE if <= 0.60 (deeper mismatch); else PARTIAL"),
    }
    out.write_text(json.dumps(res, indent=2))
    print(f"\n  gold_full {gf:.4f}  (H168 {H168_GOLDFULL}, flagship {FLAGSHIP_GOLDFULL})",
          flush=True)
    print(f"  ragtruth_en {res['ragtruth_en']['auc']}  "
          f"non-EN {res['ragtruth_nonen']['auc']}", flush=True)
    print(f"=== H169 VERDICT: {verdict} ===", flush=True)


if __name__ == "__main__":
    main()
