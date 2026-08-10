# R14-H135 (A6) - minimal-pair co-location in the admitted H108 lane: launch design

The H108 lane is the campaign's only admitted lane and its supervision is a contrast - a corrupted quantity claim against a clean claim over the same evidence chunk - that the trainer never presents. Under the flat shuffle the two members of a pair land in the same 48-row batch with probability 6.29e-5, so 0.43 of the 6,889 reconstructible pairs are co-located in an entire run. H135 packs each pair adjacent in the epoch stream and changes nothing else.

- **Registration** - `docs/experiments/semantic-grounding-experiments.md` Round 14 index row R14-H135 (A6); binding text `R14_synthesis.md` §R14-A6
- **Gates** - clause 1 PASSED before registration (6,889 paired chunks covering 29,779 of 61,184 lane rows = 48.67% vs a 30% floor); clause 2 PASSED (median pair edit similarity 0.9900, `R14_gate_H135_editsim.json`)
- **Artifacts** - trainer `R14-H135_trainer.py`, driver `R14-H135_campaign.sh`, permutation audit `R14-H135_perm_audit.json`, smoke log `logs/R14-H135_smoke.log`

## Co-location mechanism

Only batch composition changes. The loss, optimizer, schedule, mix, DANN groups, step count, seed and initialization are identical to the control arm.

- **Pair reconstruction** - the registered gate rule verbatim: first label-1 and first label-0 row per evidence chunk, in lane-file order. 6,889 pair units covering 13,778 lane rows; the trainer aborts if the count is not exactly 6,889
- **Unit shuffle** - each pair becomes one two-slot unit ordered (corrupted, clean); every unpaired row is a one-slot unit; units are shuffled together by `np.random.default_rng(seed)` - the same construction as `DR_lane_trainer.build_perm`
- **No boundary straddle** - a pair that would occupy the last slot of a 48-row batch is deferred and the next singleton ahead is swapped in; the tail-all-pairs fallback accepts a straddle and never fires here
- **Measured** - arm P(partner adjacent) 1.000000, P(partner same batch) 1.000000, 0 straddles across all 6,889 pairs; control P 0.0 with the partner a mean 5,110 optimizer steps away, against the 6.29e-5 flat-shuffle baseline (`R14-H135_perm_audit.json`)
- **Loss untouched** - no margin, no hinge, no auxiliary head; absolute score comparability is preserved by construction, which is what the H117 margin arm destroyed

Pair-unit composition, by (positive tag, negative tag): `quant_feverous`/`quant_corrupt` 4,837 (70.2%), `quant_infotabs` internal 1,724 (25.0%), `quant_scitab` internal 194, `quant_feverous` internal 134. The corruption-engine pairs dominate; the infotabs share is the same mass that carried the 0.39 median edit similarity in clause 2, so a quarter of the co-located units are chunk-mates rather than near-misses. Recorded as an adjudication caveat, not a build change - the registered pair rule is not amendable here.

## Permutation, resume and the epoch guarantee

The permutation is built once, before step 0, and persisted inside `resume.pt` beside the weights, the optimizer, the scheduler and the step counter.

- **Restart replays the same order** - a resumed run loads `perm` from the checkpoint and slices it at `start_step * BATCH`, a multiple of 48, so batch boundaries land exactly where they did in the killed run and every pair stays adjacent and unsplit
- **No rebuild on resume** - the perm is regenerated from the seed only when no `resume.pt` exists, so a mid-draw restart cannot reshuffle the stream
- **Verification** - the perm fingerprint (blake2b-64 over the index array) is printed at every start and written to `init_fingerprint.json`; the smoke run's post-resume fingerprint equals its pre-kill fingerprint
- **Epoch integrity** - one pass, every row seen exactly once, 15,560 steps, unchanged from the H108 recipe

## Control decision - FRESH SEEDED CONTROL, banked H108 draws do not substitute

The spec forbids the banked route as the primary comparison. Binding amendment (ii) of R14-A6:

> **Prefer the seeded-control route.** The banked H108 draws are unseeded in init and batch order (`R10-H108_lane.py:18`); the unpaired route widens the bar into a range the read is unlikely to reach.

and the bar clause: "against a *seeded* H108 control pair; on the unpaired-except-init route the primary widens to +0.040 per the H121 precedent."

- **Consequence** - the arm and its control are trained at the same seeds (draw 1 = 1135, draw 2 = 2135), sharing an identical init fingerprint and differing only in `perm` construction
- **Sequencing on GPU1** - arm draw 1 first, control draw 1 second. The pilot gate (draw 1 finqa >= 0.7382 AND mean >= 0.700) is stated in absolute terms and needs no control, so a failing arm draw 1 kills the block before the control's ~6 GPU-h is spent - this is the conditional the cost line "+12 if the seeded H108 control pair is bought" already prices
- **Banked draws keep one job** - they fix the numeric bar below; they are not the paired comparison

## Bar

Computed from the banked H108 PRIMARY windowed decomposed-min reads (`R10-H108_lane_draw{1,2}_windowed_result.json`).

| quantity | draw 1 | draw 2 | pair mean |
|---|---|---|---|
| finqa | 0.7291 | 0.7072 | **0.71815** |
| arena mean | 0.70618 | 0.70373 | 0.704955 |
| pubmedqa | 0.5907 | 0.5575 | 0.57410 |
| gold_full (in-domain) | 0.8589 | 0.8579 | 0.85840 |

- **PRIMARY** - finqa pair mean >= 0.71815 + 0.030 = **0.74815**, with **sign agreement on both draws** against the seeded control pair (session ruling 13 holds +0.030 as written; the judge's +0.020 fallback is the author's to take)
- **HOLD** - arena mean >= 0.7031; no subset below the H108 pair by more than 0.06, pubmedqa >= 0.5141 named explicitly; gold_full >= 0.8484
- **KILL** - finqa < 0.72815 (+0.010 over the H108 pair), or the draws disagree in sign, or the arena mean < 0.6971
- **PILOT GATE** - spend draw 2 only if draw 1 reads finqa >= 0.7382 AND mean >= 0.700
- **CONFOUND (binding)** - log-length residualization at adjudication, as A4
- **Bar tension, recorded** - 0.74815 sits above finqa's measured faithful-oracle ceiling 0.7348 under the shipped read; bars stay ceiling-blind per session ruling 6 and the H119 frozen-weight re-read banked finqa 0.7452 above it
- **Reading to resolve at adjudication** - the spec fixes the primary numerically at the banked pair + 0.030. If the fresh seeded control pair departs materially from the banked 0.71815, the adjudicator must state whether the fixed 0.74815 or `control pair + 0.030` binds; the sign-agreement clause is unaffected either way

## Mix, groups, cost

- **Mix** - 746,854 rows: clean public 685,670 + H108 lane 61,184, byte-identical to the H108 recipe
- **Groups** - 16 DANN groups: the 12 clean-mix groups plus the lane's four (`quant_corrupt` 33,176, `quant_infotabs` 16,466, `quant_feverous` 10,369, `quant_scitab` 1,173); the trainer aborts if the realised group set differs
- **Recipe** - mmBERT-base cross-encoder, BCE + DANN lambda 0.02 with the Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5, OneCycleLR 10% warmup, grad clip 1.0, 1 epoch
- **Steps** - 15,560 per draw; resume point every 1,000 steps
- **Cost** - ~6 GPU-h train + ~1 GPU-h for both reads per draw; arm d1 + control d1 ~14 GPU-h, both pairs ~28 GPU-h worst case

## Pre-launch gates

- `/dev/shm` 32G confirmed before launch (container default 64M kills the DataLoader workers)
- GPU1 idle per `nvidia-smi` at launch
- Smoke run, 50 steps at `--resume-every 20`, checkpoint redirected to `models/R14-H135-smoke`: loss finite throughout, `resume.pt` written and re-read with an identical perm fingerprint, batch contents asserted equal to the permutation slice for the first five batches, P(partner adjacent) 1.000000. Smoke artifacts deleted before the real launch

## Launch record

Arm draw 1 launched detached on GPU1 at **2026-08-10 08:09:49**, log `logs/R14-H135_campaign_d1.log`, checkpoint `models/R14-H135-arm-draw1`.

- Startup lines confirm the design: 746,854 rows / 16 domains / seed 1135, 6,889 minimal pairs, init fingerprint `229c1972a420e8916b2880869c655f12`, perm fingerprint `1227e10c9daa2922`, P(partner adjacent) 1.000000
- Step rate: 1.07 s/step averaged to step 600 (643 s); the steady-state intervals run 1.18 s/step (200 → 400) and 1.27 s/step (400 → 600), the first 200 steps being faster on short warmup batches. At 1.2-1.3 s/step the 15,560 steps project to ~5.2-5.6 h of training (ending ~13:20-13:45), with both arena reads adding ~1 h - campaign complete ~14:20-14:45. H108 draw 1 ran the same recipe at 1.39 s/step (21,695 s), which is the pessimistic bound

## Commands

```bash
# arm, draw 1 (running)
sudo -n mount -o remount,size=32G /dev/shm   # only if /dev/shm reverted to 64M
cd /home/lab/workspace/private/ai-assistants/groundrails && \
  nohup setsid bash experiments/grounding-semantic/R14-H135_campaign.sh 1 \
    >> logs/R14-H135_campaign_d1.log 2>&1 &

# seeded control, draw 1 - only after the arm clears the pilot gate
nohup setsid bash experiments/grounding-semantic/R14-H135_campaign.sh 1 control \
  >> logs/R14-H135_campaign_ctrl_d1.log 2>&1 &
```

Relaunch after any kill is the same command - the trainer resumes from `models/R14-H135-{arm,ctrl}-draw<N>/resume.pt` and replays the persisted permutation. Completion markers: `=== H135 ARM D1 CAMPAIGN COMPLETE <ts> ===` and `=== H135 CTRL D1 CAMPAIGN COMPLETE <ts> ===`. Per-draw outputs: `R14-H135_{arm,ctrl}_draw<N>_result.json` (in-domain holds) and `R14-H135_{arm,ctrl}_draw<N>_windowed_result.json` (PRIMARY blind read).
