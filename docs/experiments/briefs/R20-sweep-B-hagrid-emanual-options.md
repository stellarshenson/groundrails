# R20 sweep brief B - options for attacking hagrid and emanual

Subagent research brief (read-only, 2026-08-16). Input to the Round 20 adjudication; the coordinator adjudicates. All paths under `experiments/grounding-semantic/`.

Standing: under the incumbent's own (native) convention the two real losses are hagrid -0.1118 (ours 0.6424 vs 0.7542) and emanual -0.0914 (0.678 vs 0.7694) - `R19_corrections_census.json` B, `R19-H171_incumbent_native.json`. Under OUR harness convention we beat the incumbent on both (hagrid +0.0432, emanual +0.0781). Observed 4-draw per-subset sd: hagrid ~0.018, emanual ~0.032.

## (a) Error-mass decomposition

### hagrid (n=250, 38 negatives; flagship 2-draw 0.6424)

From `R17-H147_autopsy.json` and `R19-H162_hagrid_mechanisms.json`. Fractions of misrank mass, draw h150d1; they overlap.

| Mechanism | Mass / evidence | Fix ceiling |
|---|---|---|
| Vacuous/frame-only responses (4 items literally "Based on the given context ," labelled unsupported, scored positive at containment 0.000) | 21.2% of misrank mass (17.3% after enriched mix) | ranking them last lifts hagrid +0.076 / +0.074 (draw spread 0.0018) |
| All artifact negatives (7 items) | 27.3% of misrank mass | +0.039 |
| source_select (credit to topically-adjacent passage in multi-passage pools) | vacuous-excluded AUROC by pool depth: 1 doc 0.86/0.81, 2-3 docs 0.69/0.63, 4-8 docs 0.51/0.61; k-doc curve FALLS with added passages | h159d1 enriched checkpoint lifts the 4-8 cell +0.152, whole subset 0.6423 -> 0.7074 |
| overclaim_near_copy_reject (near-verbatim restatement asserting an unlicensed proposition) | worst-ranked unsupported items have HIGHER token containment (0.78-0.85) than supported (0.72-0.73); rejection tracks surface mismatch, not unsupportedness | causal in enriched contrast: deep-pool near-copy unsupported fall -1.193 logits (n=28), supported +0.232 |
| H147 consensus frame | negatives carry 60.5% of errors (lift 3.98); oracle ceiling faithful 0.7833 vs best checkpoint 0.6805 | headroom 0.10-0.25 |

Killed as hagrid levers: general frame stripping (gain is entirely the 4 vacuous items; costs -0.015 elsewhere), yes/no polarity (unmeasurable at 9 negatives), aggregation redesign (H147: shipped min best-in-class of 8), window geometry (anti-concentrated).

### emanual (n=132, 14 negatives; flagship 2-draw 0.678)

| Mechanism | Evidence |
|---|---|
| List/procedural register | list-structured slice 70/132 items carries 16/16 consensus errors (83% of rank loss); in-slice AUROC at CHANCE on all five banked checkpoints (0.47-0.54) while the non-list half reads 0.90-1.00 (`R17-H147_autopsy.json`) |
| discourse_frame_sink (closing recap decides the MIN) | recap sentences 3.2% of sentences, item sink 58.3% of the time (3.6x lift, mean logit -1.13 vs +1.55); 24 recap-ending items read 0.55 = chance (`R19-H162_procedural_mechanisms.json`) |
| bind_path_segment (arrowed UI path vs bare token run) | 19 items (14.4%); within-stratum AUROC 0.5833 vs 0.7107; token-multiset-identical transposition ranks 73rd percentile |
| bind_step_to_procedure (verbatim step under wrong goal heading) | 4 of 10 false positives (e.g. item 65: wrong procedure, every step verbatim, sentence scores +2.76..+3.79) - underpowered at 14 negatives |
| Lexical ceiling | plain token containment scores emanual 0.7763 vs model 0.6973 - the model is BELOW a lexical baseline here |

## (b) Where the incumbent's advantage comes from

**Convention, not model quality.** Incumbent native (question + ALL passages, ONE 4096-token pass) vs the same incumbent under our harness (question-blind, per-doc 1500-char chunks, max-over-docs): hagrid 0.7542 vs 0.5992 (+0.155 from convention alone), emanual 0.7694 vs 0.5999 (+0.169); zero items truncated; chunked replication confirms. Mapping is exact: joint pool answers hagrid's source_select by construction; question conditioning answers emanual's wrong-procedure binding (our decomposed-min read is structurally blind to it). Our own checkpoint gains the same way: H165 C0 pool-concatenated gold_full 0.9014 vs L0 per-doc 0.8659. Serving-read changes were closed by the H151 ruling - this brief prices training lanes only; the strategic fact stands: the convention gap on these two subsets is ~+0.16 each, larger than everything below combined.

## (c) Lever table

Corrected-floor arithmetic applies (pooled sd 0.01090 per `R20_variance_repair.json`). A +0.10 on one subset = +0.01 on the 10-subset mean.

| Lever | Mechanism claim | Supply | Predicted subset effect | Predicted MEAN effect | Detectability |
|---|---|---|---|---|---|
| L1 vacuous_claim_reject | frames scored supported because no mix lane shows a frame as a whole-response negative (mix census: 0 discourse frames in 685,670 clean rows vs 4.1% in hagrid) | rule-based generator, ~5-10k rows; MUST include label-1 frame+content rows so MIL learns frame-NEUTRAL (protects emanual's recap-sink items) | hagrid +0.03..+0.074; emanual recap stratum +0.01..+0.03 | +0.005..+0.010 | mean: never at sane k; verify on mechanism read (frame-only misrank 21.2% -> <5%, near-deterministic) |
| L2 source_select / attr_pool | model credits best topically-adjacent passage; no banked lane presents a choice among competing passages | BM25-distractor construction over MiniCheck (MIT, 14,356 rows, median chunk 922 chars) + VitaminC volume, document-disjoint; ~20-30k rows. Existence proof: H159 moved hagrid +0.065 | hagrid +0.05..+0.065; spillover +0.008..0.011 on 4 subsets | +0.006..+0.010 IF the H159 table collapse (finqa -0.112, tatqa -0.133, delucionqa -0.109) is avoided by isolating from FAVA/PubHealth/FinDVer | hagrid subset detectable at k=2 (bar +0.037) |
| L3 bind_product_version | model tracks identifier PRESENCE not CORRECTNESS | NVD/CVE JSON (public domain): description x CPE field-swap negatives, ~30k rows; pointer_answer_credit rides free | techqa +0.04..+0.06 | +0.004..+0.006 | techqa at k=2 (~3 seed-sd); mean never |
| L4 bind_path_segment | arrowed-path transposition negatives surface-inseparable | pure rule generator (GNOME/LibreOffice/Debian vocab), ~5-15k rows | emanual +0.02..+0.04 | +0.002..+0.004 | held-out probe only; rider, never standalone |
| L5 bind_step_to_procedure | owns emanual's list half (100% of consensus error mass) | SUPPLY BLOCKED (army-tm crawl empty of stepped manuals; multidoc2dial refuted -0.0384 R10-H107; H148 killed at gate) | up to +0.08..0.10 emanual | up to +0.010, unbuildable | n/a |

Brutal summary: no single lever reaches the k=2 mean floor. Only a PORTFOLIO arm (L1+L2+L4) predicts a mean-detectable +0.012..0.020. Repaired-H159 arithmetic caps honest upside at ~0.726 from these subsets alone.

## (d) Contamination wall

L1/L4 generators - CLEAR by construction. L2: MiniCheck + VitaminC hold GREEN R14-H136 8-gram verdicts vs all ten walled corpora; re-run the census on the NEW distractor pairings at build (VitaminC and hagrid share the Wikipedia population - entity overlap, not document overlap). L3: NVD not a RAGBench source; CVE-id overlap census + document-disjointness required at build (R10-H107 precedent). L5 moot.

## (e) Ranked recommendations

1. **Portfolio arm: L2 isolated attr_pool + L1 frame lane (+L4 riding free)** - the only mean-detectable configuration; both halves have existence proofs; the H159 failure mode is diagnosed (near-copy collateral from the OTHER four lanes). Pre-registerable: hagrid >= 0.680 per draw; frame-only misrank < 5%; hagrid k-doc slope non-negative; finqa/tatqa/delucionqa each within 1 across-seed spread of flagship; mean promotion at declared k only.
2. **L3 NVD lane** - techqa lever with the cheapest verification instrument in the arena (techqa seed sd 0.0136); mean contribution below any sane bar, techqa stratum is the instrument.
3. **L4 as rider only.**

Not recommended now: L5 (supply blocked, two prior refutations); serving-read change (closed by H151) - noted for the strategic ledger: the 0.74 target may be cheaper through the closed door (convention gap ~+0.16 x 2 subsets) than through lanes (+0.012..0.020 honest).
