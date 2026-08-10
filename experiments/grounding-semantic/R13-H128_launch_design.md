# R13-H128 - WICE-ATTRIBUTED-SUPPORT-LANE: launch design note

Pre-launch record of the implementation choices the registration left to the
builder, plus the realized lane and group tables. Written before the draw-1
launch; adjudicates nothing.

## What the registration fixes

WiCE partial-support deletion/swap negatives added to the clean public mix as a
strictness signal on over-claim, re-aimed at hagrid's 12 high-scoring negatives
(suppressing them is worth up to +0.1769). The pre-GPU kill-gate is already run
and PASSED (`R13-H128_gates_result.json`): provenance 0.000000 on all four
runs, buildable pairs 68,380 against the 15,000 bar, multi-sentence evidence
0.7077 against 0.40, licence ODC-BY 1.0 annotations over CC BY-SA Wikipedia
text. It is not re-run.

## Lane construction: the most-conservative build, 18,264 pairs

The gates costed several constructions. The one shipped here is the
**most-conservative** - `buildable_pairs_min_set_with_swap`, claim level 7,636
plus subclaim level 10,714 = **18,350** - not the 68,380 headline. Only the
SMALLEST multi-sentence evidence set of each claim is used, so no claim
contributes more than one evidence set and the near-duplicate mass of the
all-sets construction never enters the mix. The gate result documents no reason
to prefer a wider build, so the launch order's default stands.

Per pair (`R13-H128_build_lane.py`):

- **positive** - claim + the full minimal evidence set, label 1
- **deletion negative** - the same claim with one sentence of that set removed,
  label 0
- **swap negative** - the same claim with that sentence replaced by the
  lexically nearest sentence (token-set Jaccard) of a **different** article,
  label 0; 30 donorless cases fall back to deletion

Three builder decisions, all recorded because the registration does not fix
them:

1. **Evidence indices coerced** - the binding builder note from the gates
   block: claim-level `supporting_sentences` hold strings, subclaim-level hold
   ints; both are cast to int on load
2. **`not_supported` rows dropped** (20 rows, 86 negatives). A few carry a
   multi-sentence evidence set anyway, and their positive member would assert
   support for a claim WiCE annotates as unsupported. Final pairs **18,264**,
   still clear of the registered 15,000 bar
3. **`partially_supported` rows KEPT, positive labelled 1.** The lane's
   supervised contrast is presence-vs-absence of WiCE's *annotated* minimal
   evidence set, not full-claim faithfulness - `supporting_sentences` is by
   construction the minimal set that supports the annotated portion. The
   alternative (positives only from `supported` rows) yields **11,880** pairs
   and would fail the registered >= 15,000 pre-GPU gate, so it is not
   available. Recorded as the load-bearing labelling assumption of this lane:
   6,384 of the 18,264 pairs anchor on a partially-supported claim

**Exact duplicates dropped.** The positive of a k-sentence minimal set is
shared by that set's 2k negatives; emitting it 2k times would be a silent
2k-fold reweight of one (claim, chunk) pair. Positives are therefore emitted
once. **34 rows dropped as cross-label collisions** - the same claim text
appears at both annotation levels with different minimal sets, so one level's
corrupted chunk can coincide with the other level's positive; those pairs teach
nothing and are removed whole.

### Realized lane

| quantity | claim level | subclaim level | total |
|---|---|---|---|
| positives (one per claim minimal set) | 1,496 | 2,426 | **3,922** |
| deletion negatives | 3,812 | 5,320 | 9,132 |
| swap negatives | 3,801 | 5,301 | 9,102 |
| **pairs (negatives)** | 7,643 | 10,621 | **18,264** |

Lane parquet `R13-H128_lane.parquet`: **22,071 rows** after the duplicate and
collision drops, positive fraction **0.1768**, schema `claim, chunk, label,
tag` - byte-compatible with the H108 lane loader. Chunk length median 193
chars, p95 529, max 2,763; the trainer truncates to the 1,500-char serving unit
at load exactly as `R10-H108_lane.lane_train` does, so only the longest ~1% is
touched. Build log `logs/R13-H128_build_lane.log`.

## DANN group handling: one lane group, 13 total

The 12 clean-mix groups are FROZEN (the R12-H122 kill froze the group design
for the campaign) and the lane adds **one** group of its own, `wice_attrib` -
the H108/DR convention of one DANN group per lane SOURCE (H108 added four for
four source corpora, the DR lane four for four generation arms). WiCE is a
single source under a single construction engine, so it takes a single group.
The rejected alternative was splitting claim-level and subclaim-level into two
groups: same Wikipedia text, same engine, only claim granularity differs, so
the split would hand the discriminator a distinction that is not a register.

| group | rows |
|---|---|
| halueval | 40,000 |
| psiloqa | 61,712 |
| ragtruth_cn / de / en / es / fr / hu / it / pl | 15,090 each |
| tabfact | 92,585 |
| vitaminc | 370,653 |
| **wice_attrib** | **22,071** |
| **MIX TOTAL** | **707,741** |

`n_groups` = **13**, chance 0.077. Per-group discriminator accuracy over the
final 20% of the epoch is written to the result json
(`domain_acc_per_group_final20pct`), the H127 amendment-4 convention, so the
GRL confound stays visible beside the blind read.

## Mix, steps and recipe

Clean public mix via `R10-H108_lane.public_train()` (685,670 rows) plus the
lane (22,071) = **707,741 rows**, **14,745 steps** at BATCH 48, one epoch.
Recipe unchanged from the clean R9-H105 control: mmBERT-base cross-encoder, BCE
+ DANN lambda 0.02 under the Ganin ramp, MAX_LEN 512, BATCH 48, LR 1e-5,
OneCycleLR one epoch, grad clip 1.0. Realized step rate on GPU1 is **1.16
s/step** measured over steps 200-400 of draw 1 (H127's whole-epoch rate was
1.258 s/step), so a draw trains in ~4.9 h and the full campaign lands in ~5.6 h
with both reads - draw 1 completion expected ~08:00 on 2026-08-10.

## Seeding and pairing

Draw seeds **{1: 1128, 2: 2128}**. Per session ruling 8 / H126,
`torch.manual_seed(seed)` is issued before model construction and **re-issued
immediately after**, before any dropout or forward. The trunk+task_head
blake2b-128 init fingerprint is written to
`models/R13-H128-draw<N>/init_fingerprint.json` (draw 1:
`e22b71c1562be44b97ae5a9a1f68f4aa`, reproduced across two independent probe
runs).

**Pairing caveat (load-bearing, for the adjudicator).** The controls are the
**banked clean draws**, which are unseeded (pre-H126), so the realized
comparison is arm-vs-banked-control, not init-paired - as in R12-H122 and
R13-H127. It is weaker here than in H127: `n_groups` is 13 against the
control's 12, so the domain head has a different shape and construction
consumes a different RNG draw. The registered clauses should be read against
that weaker pairing; the fingerprint is written so a future seeded 13-group
control at the same seeds can be checked against it.

## Bars (registered, unmodified)

- **Primary** - hagrid **>= 0.688** (+0.040 on the incumbent, ~4 SE),
  subset-primary per ruling 7
- **Hold** - mean **>= 0.7031** (the clean 2-draw control mean); finqa and
  techqa hold per ruling 9
- **1-draw pilot gate** - mean **>= 0.700** AND hagrid **>= +0.02**; BOTH are
  required to spend draw 2. Either missing and draw 2 stays unspent
- Reads: the PRIMARY read is the windowed decomposed-min blind arena read
  (1,500-char windows, stride 750); the truncated decomposed read runs first
  for comparability, and the in-domain holds (gold, gold_full, ragtruth_en,
  ragtruth_nonen) come out of the trainer's own evaluate pass

## Artifacts and relaunch

- Lane builder `R13-H128_build_lane.py` -> `R13-H128_lane.parquet`
- Trainer `R13-H128_trainer.py --draw {1,2}` (`--max-steps N` for probe runs;
  mid-run `resume.pt` every 1,000 steps, atomic replace, epoch permutation
  persisted with the weights)
- Campaign `R13-H128_campaign.sh <draw>` - train + in-domain holds ->
  truncated decomposed read -> windowed read (PRIMARY); final marker
  `=== H128 D<N> CAMPAIGN COMPLETE <timestamp> ===`
- Results `R13-H128_draw<N>_result.json`,
  `R13-H128_draw<N>_windowed_result.json`; checkpoint
  `models/R13-H128-draw<N>/`; log `logs/R13-H128_campaign_d<N>.log`

**Relaunch after any interruption** (the trainer resumes from `resume.pt`; a
container restart reverts `/dev/shm` to 64M and must be remounted first):

```
sudo -n mount -o remount,size=32G /dev/shm
cd /home/lab/workspace/private/ai-assistants/groundrails && \
  nohup setsid bash experiments/grounding-semantic/R13-H128_campaign.sh 1 \
  >> logs/R13-H128_campaign_d1.log 2>&1 &
```

Draw 1 launched **2026-08-10 02:22:32**, PID 559831, GPU1.
