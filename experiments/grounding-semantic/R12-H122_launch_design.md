# R12-H122 - DANN-GROUP-COLLAPSE: launch design note

Pre-launch record of the one design question the registration left open, resolved
per session ruling 1. Written before the draw-1 launch; adjudicates nothing.

## The question

The registration (`R12_synthesis_full_field.md`, R12-H122) states the merge as
**16 → 9** and its amendment A3 pins the control as "the H108-admitted 16-group
recipe at 746,854 rows". Session ruling 1 (canonical log, 2026-08-08) pinned the
R12 incumbent differently: **the CLEAN recipe at 0.7031 (H105 pair), frozen at
registration, no re-baselining mid-round**. The two are incompatible - A3's
16-group mix is clean + H108, which ruling 1 does not admit as the incumbent.

Ruling 1 governs (it is later, explicit, and round-wide). The arm therefore runs
on the **clean public mix**: 685,670 rows, `R10-H108_lane.public_train()`,
**12** DANN groups - not 16.

## Merge logic transferred to the clean mix

The registered merge principle is a tag→index remap only: the eight RAGTruth
language tags become one tag; rows, labels, sampling order, natural frequency,
lambda, ramp and schedule are unchanged. Applied to 12 groups this is 12 - 7 = **5**.

| clean tag | H122 group |
|---|---|
| `ragtruth_en` | `ragtruth` |
| `ragtruth_de` | `ragtruth` |
| `ragtruth_fr` | `ragtruth` |
| `ragtruth_es` | `ragtruth` |
| `ragtruth_it` | `ragtruth` |
| `ragtruth_pl` | `ragtruth` |
| `ragtruth_hu` | `ragtruth` |
| `ragtruth_cn` | `ragtruth` |
| `halueval` | `halueval` |
| `psiloqa` | `psiloqa` |
| `vitaminc` | `vitaminc` |
| `tabfact` | `tabfact` |

**Resolved: 12 → 5 groups** (chance 0.200 vs 0.083). Domain-head output layer
changes by 7 × (768 + 1) = 5,383 parameters of 307.1M; nothing else moves.

**Gate evidence transfers.** The licence (`R12-H122_gradgate_result.json`) measured
the 16-way vs 9-way GRL trunk-gradient ratio at 1.1869 (bar ≥ 1.15) and direction
cosine 0.0254 (bar ≤ 0.9) on frozen `models/R10-H108-lane-draw1`. Both the gated
and the launched merge remove the same seven redundant language tags from the same
adversarial label space; the measured quantity is the trunk gradient the RAGTruth
language split contributes, which is identical in the clean mix. The gate was not
re-run at 12-vs-5 - recorded as a caveat, not re-adjudicated here.

## Control and pairing

Controls are the **banked clean draws** per session ruling 8: no new control
training. Windowed pair mean **0.7031** (`R9-H105_windowed_result.json` 0.7047,
`R9-H105_draw2_windowed_result.json` 0.7015); `gold_full` 0.8788 / 0.8240 (pair
0.8514); `ragtruth_nonen` 0.8402 / 0.8337 (pair 0.83695).

**Pairing caveat (load-bearing, for the adjudicator).** The registration calls this
design "genuinely paired" because row and step count are unchanged, so batch order
is shareable. The banked controls are **unseeded** (pre-H126), so the realised
comparison is arm-vs-banked-control, not init-paired. Amendment A2's
"assert bit-identical trunk+task_head init in both arms" cannot bind against an
unseeded control; the arm still writes `init_fingerprint.json` so a future seeded
12-group control can be checked against it. The registered sign-agreement clause
should be read against this weaker pairing.

## Bars carried forward (registered, re-priced onto the clean control)

- **Improve** - pair mean ≥ **0.7091** (control 0.7031 + 0.006) with sign agreement on both draws
- **KILL** - pair mean < **0.7051** (control + 0.002), or the paired draws disagree in sign
- **Hold** - `ragtruth_nonen` ≥ **0.81695**; `gold_full` ≥ **0.8414**; no blind subset < 0.55
- **Attribution** - no subset concentration predicted; a single subset carrying |delta| > 0.05 and supplying the whole mean move FALSIFIES the attribution (recorded unattributed)
- **Counter-prior (A4)** - R8-H93 measured LOCO transfer rising monotonically with invariance pressure; merging groups reduces that pressure, so a null reads as "the H93 direction stands", not as noise

## Seeding (A2 trap / H126)

`n_groups` changes the domain head's RNG consumption at construction, which would
desync every subsequent dropout mask. `torch.manual_seed(seed)` is issued before
construction and **re-issued immediately after**, before any forward or dropout;
`task_head` is constructed before `domain_head`, so trunk + task_head init is
unaffected by the group count either way. Fingerprint (blake2b-128 over
trunk + task_head) written to `models/R12-H122-draw<N>/init_fingerprint.json`.
Draw seeds **{1: 1122, 2: 2122}**. The row permutation uses
`np.random.default_rng(seed)` and is independent of the torch stream.

## Artifacts

- Trainer `experiments/grounding-semantic/R12-H122_trainer.py` (clean recipe: BCE + DANN λ0.02 Ganin ramp, OneCycleLR 1 epoch, MAX_LEN 512, BATCH 48, LR 1e-5; mid-run `resume.pt` every 1,000 steps)
- Campaign `experiments/grounding-semantic/R12-H122_campaign.sh <draw>` - train → truncated read → windowed read (PRIMARY)
- Results `R12-H122_draw<N>_result.json`, `R12-H122_draw<N>_windowed_result.json`; checkpoint `models/R12-H122-draw<N>/`
