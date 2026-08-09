# R13-H127 - RAGTRUTH-PARALLEL-COPY-REBALANCE: launch design note

Pre-launch record of the implementation choices the registration left to the
builder, plus the realized group table. Written before the draw-1 launch;
adjudicates nothing.

## What the registration fixes

Family-mass-preserving reweight of the RAGTruth family: `ragtruth_en` per-row
weight **4.0**, each of the seven translations **0.5714**, family total held at
its natural row-equivalent mass, every other corpus untouched. The CPU
alignment gate is already run and PASSED (label agreement 0.9998-1.0000 on all
seven, `task_type` agreement 1.0000, `pos_frac` spread 0.000199 against a
<0.002 gate, numeric-token Jaccard 0.84-0.88 aligned vs 0.13 shuffled) - it is
not re-run.

## Implementation choice: sampling weights as an integer row multiset

The weights are realized as a **row-multiplicity index multiset over the
unchanged mix arrays**, not as a `WeightedRandomSampler` and not as a loss
reweight. `WeightedRandomSampler` draws with replacement from a stateful
generator - neither deterministic under mid-epoch resume nor exactly
mass-preserving in a finite epoch. The multiset is both.

- **EN** - each `ragtruth_en` row index is emitted exactly 4 times; the
  realized per-row weight is 4.0 exactly, no rounding
- **Translations** - each contributes a seeded without-replacement subsample.
  The residual family budget (family mass minus the EN mass) is split across
  the seven by **largest-remainder allocation** in proportion to natural size,
  ties broken on the fixed order de, fr, es, it, pl, hu, cn. The subsample
  draw uses `np.random.default_rng([seed, 127])`, a stream independent of the
  batch-order permutation
- **Everything else** - multiplicity 1, asserted equal to its natural count
- **Order** - the multiset is permuted once by `np.random.default_rng(seed)`
  and stored in `resume.pt` alongside the step counter, so a resumed run
  replays the identical stream. Checkpoint every 1,000 steps, atomic replace

Because `4.0 + 7 x 4/7 = 8.0` exactly and all eight files carry the same
15,090 rows, the family mass, the total mix row count and the step count are
**unchanged from the control** - the arm is step-for-step paired at 14,285
steps.

## Realized per-group row-equivalents (measured at launch)

| group | natural | reweighted | per-row weight |
|---|---|---|---|
| halueval | 40,000 | 40,000 | 1.0000 |
| psiloqa | 61,712 | 61,712 | 1.0000 |
| **ragtruth_en** | 15,090 | **60,360** | **4.0000** |
| ragtruth_de | 15,090 | 8,623 | 0.5714 |
| ragtruth_fr | 15,090 | 8,623 | 0.5714 |
| ragtruth_es | 15,090 | 8,623 | 0.5714 |
| ragtruth_it | 15,090 | 8,623 | 0.5714 |
| ragtruth_pl | 15,090 | 8,623 | 0.5714 |
| ragtruth_hu | 15,090 | 8,623 | 0.5714 |
| ragtruth_cn | 15,090 | 8,622 | 0.5714 |
| tabfact | 92,585 | 92,585 | 1.0000 |
| vitaminc | 370,653 | 370,653 | 1.0000 |
| **RAGTruth family** | **120,720** | **120,720** | - |
| **MIX TOTAL** | **685,670** | **685,670** | - |

Family share of the public mix is unchanged at 17.6%; the EN share inside the
family moves 12.5% → 50.0%.

**Registration arithmetic slip recorded.** The registration states the family
at **120,717** row-equivalents. The live mix measures **120,720** - eight files
of exactly 15,090 rows each, identical pre- and post-filter (`context` /
`prompt` length > 50 removes nothing). The +3 is a registration slip, not mix
drift. The trainer therefore enforces mass preservation against the **measured
natural total** (realized within ±1) and separately aborts if the measured
total departs from the registered 120,717 by more than 4, which is the drift
guard. Nothing about the mechanism or the bars changes.

## Mix, groups and recipe

Clean public mix only via `R10-H108_lane.public_train()` - 685,670 rows, no DR
lane, no H108 lane. **DANN groups stay at 12**: the R12-H122 kill froze the
group design for the campaign, so the eight RAGTruth language tags remain eight
separate adversarial classes. Recipe unchanged from the clean R9-H105 control:
mmBERT-base cross-encoder, BCE + DANN λ 0.02 under the Ganin ramp, MAX_LEN 512,
BATCH 48, LR 1e-5, OneCycleLR one epoch, grad clip 1.0.

## Seeding and pairing

Draw seeds **{1: 1127, 2: 2127}**. Per session ruling 8 / H126,
`torch.manual_seed(seed)` is issued before model construction and **re-issued
immediately after**, before any dropout or forward. `n_groups` is 12 here,
identical to the control, so construction consumes the same RNG either way; the
re-issue is kept so the arm's stream matches any future seeded 12-group
control. The trunk+task_head blake2b-128 fingerprint is written to
`models/R13-H127-draw<N>/init_fingerprint.json` (draw 1:
`b23ee8b80a1bd724b6c1c616fd18ef45`).

**Pairing caveat (load-bearing, for the adjudicator).** The controls are the
**banked clean draws**, which are unseeded (pre-H126). The realized comparison
is therefore arm-vs-banked-control, not init-paired, exactly as in R12-H122.
The registered sign-agreement clause should be read against this weaker
pairing. The fingerprint is written so a future seeded 12-group control at the
same seeds can be checked against it.

## Bars (registered, unmodified)

- **ADMIT** - 2-draw pair blind windowed mean ≥ **0.7150** with sign agreement
  on both draws, and all holds: `ragtruth_nonen` ≥ **0.82** on both draws,
  `gold_full` ≥ **0.84**, no arena subset < **0.55**
- **REFUTE** - pair mean < **0.70496** (the H108 incumbent) or sign
  disagreement across draws

**Control-reference note recorded, not re-adjudicated.** The registered bar is
priced against the H108 incumbent 0.70496 while this arm trains on the clean
public mix, whose banked control pair is 0.7031. Both numbers are carried into
the adjudication as written; the coordinator pinned the bars, and re-pricing
mid-round is not a builder decision.

## Amendment compliance

1. Alignment is **positional** - recorded as positional-index alignment
   corroborated by the numeric-Jaccard control; no index-keyed join is claimed
   and none is performed (the translation subsamples are drawn independently
   per language, so no cross-language row correspondence is relied on)
2. The non-EN kill-gate is struck; `ragtruth_nonen` ≥ 0.82 stands as a HOLD
   only, a breach reported as a deliverable finding
3. The EN-below-non-EN inversion is absent from the motivation; the lever
   stands on parallel-copy redundancy alone
4. The intervention changes per-group DANN mass (EN group 4x), so the trainer
   accumulates **per-group discriminator accuracy over the final 20% of the
   epoch** and writes it to the result json as
   `domain_acc_per_group_final20pct`, reported beside the blind read so the GRL
   confound is visible. **Not co-run or cross-adjudicated with R12-H122**
   (H122 is killed and its group merge is not in this build)
5. Shares are restated above against the mix actually served here (685,670
   clean public rows, family 17.6%); the registration's 16.2% is against the
   746,854-row H108 mix, which this arm does not use
6. Seeding per R3/H126 as described above

## Artifacts

- Trainer `experiments/grounding-semantic/R13-H127_trainer.py --draw {1,2}`
  (`--max-steps N` for probe runs; mid-run `resume.pt` every 1,000 steps,
  atomic replace)
- Campaign `experiments/grounding-semantic/R13-H127_campaign.sh <draw>` -
  train → truncated decomposed read → windowed read (PRIMARY); final marker
  `=== H127 D<N> CAMPAIGN COMPLETE ===`
- Results `R13-H127_draw<N>_result.json`,
  `R13-H127_draw<N>_windowed_result.json`; checkpoint
  `models/R13-H127-draw<N>/`; log `logs/R13-H127_campaign_d<N>.log`
- Relaunch after any interruption: the identical command, the trainer resumes
  from `resume.pt`
