# R14 Evidence E4 - item-level forensics on finqa and delucionqa

**Question**: at item level, what does the clean model actually get wrong on finqa and delucionqa?

**Artifact**: `experiments/grounding-semantic/R12-H121_gateA_scores.parquet` - 77,171 sentence-by-window rows scored by the frozen R9-H105 draw-1 clean checkpoint (window 1500, stride 750).
**Scripts**: `R14_E4_item_forensics.py` (recompute + discordance decomposition), outputs `R14_E4_item_forensics.json` and `R14_E4_worst_items.txt` (verbatim dump, 638 lines).
**Discipline**: ANALYSIS ONLY. Polars throughout. No arena quantity below may enter a lane's size, thresholds or mix.

---

## Summary

Two different diseases wearing the same symptom.

- **finqa is an OBJECTIVE problem** - the subset's gold calls a correctly *derived* number "supported"; the model is trained to score *literal* entailment. 362 of 563 finqa scored sentences (64.3%) assert at least one number that appears in no retrieved window, and 75.7% of those are gold-supported. Those sentences score 0.487 against 0.641 for sentences whose numbers are all literally present
- **delucionqa is a READ problem** - the min-over-sentences rule hands the response score to a sentence that carries no checkable proposition. On 53 of 172 supported delucionqa responses (30.8%) the deciding sentence was never keyed by the annotator at all. Mean-over-sentences scores 0.8445 against the shipped min's 0.7975 on the same frozen scores
- **Neither is a capacity problem.** The same model, same window, same table: "The long term debt obligations were \$70,842" scores 0.9811 while "the total scheduled maturities of long term debt were \$77,724" scores 0.0822 - both rows are verbatim in the window. Discrimination is present and sharp; it is aimed at the wrong target
- **Both AUROCs are 3-to-5-item measurements.** Dropping the 5 worst-ranked unsupported responses moves finqa 0.6489 → 0.8012 and delucionqa 0.7975 → 0.9394. finqa has 20 unsupported responses in 250; delucionqa has 12 in 184
- **Label noise is real but small; label *absence* is large.** 11 of 20 finqa non-adherent responses have zero sentences keyed unsupported by the same annotation pass - the response-level label there is scoring answer correctness, not sentence grounding. Honest estimate of flip-worthy gold: 1-3 of 20 on finqa (5-15%), ~1 of 12 on delucionqa (8%)

---

## 1. Reproduction of the banked clean read

Sentence score = max over its windows; response score = min over its sentences; per-subset metric = response-level AUROC.

| subset | recomputed AUROC | banked R9-H105 windowed |
|---|---|---|
| covidqa | 0.8030 | 0.8030 |
| delucionqa | 0.7975 | 0.7975 |
| emanual | 0.6883 | 0.6883 |
| expertqa | 0.7857 | 0.7857 |
| finqa | 0.6489 | 0.6489 |
| hagrid | 0.6259 | 0.6259 |
| hotpotqa | 0.6809 | 0.6809 |
| pubmedqa | 0.6201 | 0.6201 |
| tatqa | 0.7034 | 0.7034 |
| techqa | 0.6934 | 0.6934 |
| **mean** | **0.70471** | **0.70471** |

Exact to 4 dp on every subset against `R9-H105_windowed_result.json`. The item-level analysis below is therefore on the same numbers the campaign banks.

---

## 2. Base rates, distributions, error concentration

### Base rates and separation

| | finqa | delucionqa |
|---|---|---|
| responses | 250 | 184 |
| supported (adherent, label 1) | 230 (92.0%) | 172 (93.5%) |
| unsupported (label 0) | 20 | 12 |
| scored sentences | 563 | 929 |
| sentences per response | 2.25 mean, 7 max | 5.05 mean, 12 max |
| sentence labels: unsupported / supported / unkeyed | 13 / 485 / 65 | 14 / 548 / 367 |
| response score, supported | 0.4291 (sd 0.190) | 0.7145 (sd 0.245) |
| response score, unsupported | 0.3410 (sd 0.193) | 0.4052 (sd 0.270) |
| separation of response means | 0.088 | 0.309 |
| sentence score, gold-supported | 0.5377 | 0.8850 |
| sentence score, gold-unsupported | 0.4800 | 0.6429 |
| **sentence-level AUROC** | **0.5870** (n1=485, n0=13) | **0.7767** (n1=548, n0=14) |

finqa is not merely mis-ranked - it is *compressed*. Supported financial sentences average 0.538; the model is broadly unconvinced by evidence it can see. delucionqa's sentence-level signal is healthy (0.885 vs 0.643) and the response-level number is worse than the sentence-level number, which is the signature of a read problem rather than a scoring problem.

### Where the discordance sits

AUROC loss decomposes into discordant (supported, unsupported) pairs. Each pair is counted once from each side.

| | finqa | delucionqa |
|---|---|---|
| total pairs | 4,600 | 2,064 |
| discordant pairs | 1,615 | 418 |
| top-5 **unsupported** items' share | **57.5%** (uniform would be 25%) | **82.5%** (uniform would be 41.7%) |
| top-5 **supported** items' share | 6.0% | 12.2% |
| supported items with zero discordance | 13 / 230 | 44 / 172 |

**The failure is false-negative dominated and item-concentrated.** No supported item carries much loss on its own (worst finqa supported item loses to 20 of 20 unsupported - 1.2% of the total); the unsupported side is where the mass is.

Leave-out ladder:

| drop worst k unsupported | finqa | delucionqa |
|---|---|---|
| 0 | 0.6489 | 0.7975 |
| 1 | 0.6801 | 0.8467 |
| 2 | 0.7138 | 0.8837 |
| 3 | 0.7512 | 0.9141 |
| 5 | 0.8012 | 0.9394 |

finqa's ladder: resp 200 (13.4% of all discordance), 71 (13.2%), 67 (13.1%), 229 (10.4%), 114 (7.4%) - five items are 57.5%.
delucionqa's ladder: resp 65 (30.6%), 82 (21.5%), 24 (16.0%) - three items are 68.2%.

That the *supported* side is diffuse and the *unsupported* side concentrated does not mean supported items are fine. It means each supported item's low score costs little because there are only 20 (12) unsupported items to be outranked by. In absolute terms 217 of 230 finqa supported responses lose to at least one hallucinated response.

### The read rule itself

| read (same frozen scores) | finqa | delucionqa |
|---|---|---|
| min over sentences (**shipped**) | **0.6489** | **0.7975** |
| mean over sentences | 0.5807 | 0.8445 |
| min over annotator-keyed sentences only (diagnostic) | 0.6159 | 0.7987 |

The min rule is *right* for finqa (+0.068 over mean) and *costly* on delucionqa (-0.047 against mean). Note that this comparison is measured on the arena and cannot itself justify a read change - a mean read would have to be argued independently and pre-registered blind.

---

## 3. Verbatim item reading and error categories

Read verbatim: 15 worst supported + 15 worst unsupported on finqa (30 items), 15 worst supported + all 12 unsupported on delucionqa (27 items). Full text with the argmin sentence and its best-scoring window in `R14_E4_worst_items.txt`.

### finqa - supported responses scored low (15 read)

| category | n | items |
|---|---|---|
| numeric derivation / arithmetic - asserted value appears in no window | 8 | 147, 205, 112, 101, 50, 184, 195, 148 |
| table column indexing - value present, must be selected by year column | 3 | 160, 96, 155 |
| cross-window aggregation - inputs exist but not in the argmax window | 2 | 217, 191 |
| numeric surface form - value verbatim in window, still scored ~0 | 1 | 5 |
| absence / no-change claim | 1 | 176 |
| ambiguous or noisy gold | 0 | - |

**resp 5 (score 0.0822) - the diagnostic case.** Both sentences read the same 266-char table window:

> SENT (0.0822): "As of December 31, 2012, the total scheduled maturities of long term debt were \$77,724."
> SENT (0.9811): "The long term debt obligations were \$70,842."
> WINDOW: `[["2012","$ 6882"], ["2013 ( 1 )","65919"], ..., ["total scheduled maturities of long term debt","77724"], ["less current maturities of long term debt","-6882 ( 6882 )"], ["long term debt obligations","$ 70842"]]`

Both target values are literally present. The one the model rejects is the cell serialized without a `$` sign; the one it accepts carries `$ 70842`. Verified by string search: `77724` and `70842` are both in the response's window pool. This is surface-form brittleness on the *same* table, not an inability to read the table.

**resp 147 (0.0506) - the objective mismatch in its purest form.**

> SENT (0.0506): "The net change in Aon 2019's unpaid restructuring liabilities during 2007 was a decrease of \$71 million."
> SENT (0.8943): "This is calculated by adding the amount expensed in 2007 (\$38 million) to the cash payments made in 2007 (-\$110 million) and adding the foreign currency revaluation (\$1 million)."
> WINDOW: `..., ["expensed in 2007","38"], ["cash payments in 2007","-110 ( 110 )"], ["foreign currency revaluation","1"], ["balance at december 31 2007","$ 63"]`

`71` appears nowhere in the evidence (verified). The *derivation* sentence, whose numbers are all present, scores 0.8943. The *conclusion* sentence, arithmetically correct, scores 0.0506. Gold marks both supported. The min read then hands the response 0.0506. The model is behaving exactly as an entailment scorer should; RAGBench-finqa is asking a different question.

**resp 160 (0.0803) - unit conversion plus column indexing.**

> SENT (0.0803): "For 2006, the residential mortgage loan balance was \$6.3 billion." — WINDOW: `["residential mortgage","9557","6337"]`
> SENT (0.9731): "For 2007, the residential mortgage loan balance was \$9.6 billion." — WINDOW (prose): "...\$ 3.0 billion of the \$ 9.6 billion of residential mortgage loans were interest-only loans"

The 2007 figure exists as the literal string `9.6 billion` in prose → 0.9731. The 2006 figure requires reading column 2 of a table in millions and rounding 6337 → \$6.3 billion → 0.0803. Same claim template, same document set, 0.89 apart.

**resp 217 (0.0546) and resp 191 (0.1376) - cross-window.** For 217, `46.7` *is* present in the response's window pool but not in the window that scored highest; the argmax landed on allowance-policy prose. For 191, the three addends `172`, `179`, `147` are present and the sum `498` is not.

### finqa - unsupported responses scored high (15 read)

| category | n | items |
|---|---|---|
| arithmetic / relational - numbers verbatim, relation wrong or derived | 6 | 200, 114, 168, 41, 189, 157 |
| temporal misbinding - right table, wrong year asserted | 2 | 36, 229 |
| lexical-overlap trap - all tokens present, entity or unit binding wrong | 2 | 71, 48 |
| meta / abstention claim ("the passage does not mention X") | 1 | 85 |
| speculation / hypothetical extrapolation | 1 | 215 |
| self-contradiction / direction reversal | 1 | 31 |
| conflicting evidence across documents (ambiguous gold) | 1 | 242 |
| probable label noise | 1 | 67 |

**resp 200 (0.7493, 13.4% of all finqa discordance) - the top false negative.**

> SENT (0.7493, gold unsupported): "Ratio = \$6.2 billion / \$38.8 billion = 0.16. So the ratio of the net increase in securities sold under agreements to repurchase to the net transfers in is 0.16."
> WINDOW: "...the increase in securities sold under agreements to repurchase of \$5 billion is driven by a \$6.2 billion increase from net transfers in... the increase in long-term debt of \$2.2 billion is driven by: the net transfers in of \$38.8 billion..."

Both operands are verbatim in the window. They belong to *different line items* - \$6.2bn is repurchase-agreement net transfers, \$38.8bn is long-term-debt net transfers - so the ratio is a category error. The model sees two exact numeric matches and a fluent relational frame and scores 0.75. This is the canonical quantitative near-miss: the tokens are all correct and the binding is not.

**resp 67 (0.7087) - the clearest label-noise candidate.**

> SENT (0.7087, sentence gold **supported**, response gold **non-adherent**): "The portion of total smokeless products shipments related to the Copenhagen segment during 2014 is 448.6 million cans and packs."
> WINDOW: `["copenhagen","448.6","426.1","392.5"], ..., ["total smokeless products","793.3","787.5","763.3"]`

The claim is literally correct and the annotator's own sentence pass marks it supported. The response-level label is 0, presumably because the question asked for a *portion* (448.6/793.3 = 56.5%) and the answer gave the raw count. That is an answer-correctness judgement, not a grounding judgement. The model gets it "wrong" only against the former.

**resp 36 (0.3815) - temporal misbinding.** Verbatim-near-duplicate of resp 5's claim but with the wrong year ("as of December 31, 2011 was \$77,724,000" against a 2012 table). All three sentences are gold-supported at sentence level; the response is gold non-adherent. The model has no year-anchoring mechanism and scores it mid-range.

**resp 85 (0.3366) - the abstention claim.**

> SENT (0.3366, gold unsupported): "However, the passage does not explicitly mention the pre-tax earnings for 2016."
> A sibling window contains: "pre-tax earnings were \$1.42 billion in 2017, 25% higher than [2016]"

The claim is a meta-statement *about* the document's coverage. A pairwise sentence-vs-window entailment scorer has no representation for "no window contains X" - it can only score this sentence against one window at a time, and a window that does not mention 2016 pre-tax earnings is weak evidence *for* the claim as much as against it. Structurally unreachable under the current read.

**resp 215 (0.3346) - speculation.** "If we assume a similar improvement in 2011 compared to 2010, we can expect the operating ratio in 2011 to be around 65.1%." Hypothetical, gold-unsupported, and the arithmetic is internally consistent with the evidence (70.6 - 5.5 = 65.1), so the derivation sentence scores 0.7544.

**resp 31 (0.2900) - self-contradiction.** "Total shareholders' equity decreased by \$2,749 million from 2012 to 2013. (Decrease from \$75,716 million in 2012 to \$78,467 million in 2013)." The parenthetical describes an *increase*. Both sentences are gold-supported at sentence level; only the whole-response label catches it. No sentence-local scorer can.

### delucionqa - supported responses scored low (15 read)

| category | n | items |
|---|---|---|
| paraphrase blindness / inference gloss | 6 | 6, 72, 137, 171, 100, 180 |
| summary or aggregation sentence spanning multiple windows | 3 | 174, 102, 105 |
| **read artifact** - min falls on contentless discourse glue or a list-header stub | 3 | 164, 114, 121 |
| partial conjunct / added detail | 2 | 55, 116 |
| argmax window miss - the supporting window exists and scored lower | 1 | 103 |
| ambiguous or noisy gold | 0 | - |

**resp 164 (0.0139) - the read artifact.** A 12-sentence response, 7 gold-supported, 5 never keyed. The deciding sentence:

> SENT (0.0139, **unkeyed**): "This helps maintain good visibility and air circulation."

There is no proposition here to ground. It is discourse glue produced by the generator; the annotator did not key it because there is nothing to key; the min read makes it the entire response score. Same shape at resp 114 ("Based on the information provided, here are the key steps to properly warm up the engine in cold weather: 1." - score 0.2492, unkeyed) while the substantive sentence in the same response scores 0.8575.

**resp 6 (0.0755) - paraphrase blindness.**

> SENT (0.0755, gold supported): "This measurement should be taken when the trailer is fully loaded and ready to be towed."
> WINDOW: "The recommended way to measure GTW is to put your fully loaded trailer on a vehicle scale... in its 'loaded and ready for operation' condition."
> Sibling sentence, near-verbatim quotation of the same window: 0.9636

The paraphrase "fully loaded and ready to be towed" ↔ "loaded and ready for operation" costs 0.89. The model is a near-verbatim matcher on this subset - measured: score correlates 0.55 with content-word overlap between the sentence and its best window (gold-supported sentences average 0.798 overlap; gold-unsupported average 0.571).

**resp 103 (0.2385) - argmax miss.** The sentence "the illuminated entry system (Headlight Illumination On Approach) is activated when the vehicle is unlocked using the Passive Entry system" is directly supported by the window at doc 1 offset 0 ("Passive Entry Unlock initiates Headlight Illumination On Approach"), but the highest-scoring window was doc 1 offset 2250, which does not contain that sentence. The max over windows picked wrong.

### delucionqa - all 12 unsupported responses read

| category | n | items |
|---|---|---|
| lexical-overlap trap / partial-conjunct fabrication | 4 | 65, 83, 93, 82 |
| relevance-not-entailment - verbatim quotation, wrong question | 2 | 82, 24 |
| read artifact / label absence - min lands on a supported sentence | 3 | 130, 106, 175 |
| subtle paraphrase drift in a procedural step (arguable gold) | 1 | 77 |
| correctly caught, negligible discordance | 3 | 56, 76, 168 |

(resp 82 is counted in two rows - its unsupported sentence is both verbatim and off-topic.)

**resp 65 (0.9152, 30.6% of all delucionqa discordance) - the single most expensive item in the subset.** One sentence, so min = max:

> SENT (0.9152, gold unsupported): "The TrailCam image can be deactivated by pressing the touchscreen X button, shifting the transmission into PARK, turning the ignition OFF, or activating the windshield washing process."
> WINDOW: "...the TrailCam image will be displayed continuously until deactivated via the touchscreen X button, the transmission is shifted into PARK, or the ignition is placed in the OFF position... The Clean Camera system is not available when windshield washing is in process."

Three of four conjuncts are verbatim. The fourth inverts the manual: windshield washing *disables the camera-wash feature*, it does not deactivate the TrailCam image. Every content word in the fabricated conjunct appears in the window. A max-pooled entailment score over a 1500-char window has no mechanism to penalise one bad conjunct in a four-conjunct sentence that is otherwise a quotation.

**resp 24 (0.6866) and resp 82 (0.7911) - relevance, not entailment.**

> resp 24 SENT (0.6866, gold unsupported): "However, it is mentioned that FCA US LLC does not recommend deactivating BeltAlert, so it is important to check the specific instructions for your vehicle."
> WINDOW ends: "BeltAlert can be activated or deactivated by an authorized dealer. FCA US LLC does not recommend deactivating BeltAlert."

The sentence is a *literal quotation of the retrieved evidence*. It is gold-unsupported because the question was about Daytime Running Lights and BeltAlert is a different system that happened to be in the retrieved chunk. Same at resp 82, whose gold-unsupported sentence ("ensure that the hood is fully latched before driving") scores 0.9546 and is verbatim in the window. **No entailment scorer can produce a low score for a verbatim quotation.** These two items are 37.5% of delucionqa's discordance and they are not entailment failures at all - they are relevance failures, and the arena's response-level label conflates the two.

**resp 175 (0.2829) - correct detection diluted by the read.** The response contains a fabricated quote ("The context stresses the importance of 'ALWAYS driving safely with your hands on the steering wheel and obeying all applicable laws'" - not in any retrieved window). The model scores that sentence 0.6631, but a *gold-supported* sentence in the same response scores 0.2829 and becomes the min. The read discarded the model's own detection.

---

## 4. Label-noise share among "errors" - honest estimate

Three distinct phenomena must not be conflated.

**(a) Outright wrong gold - small.**
finqa: 1 clear case in 15 read (resp 67 - claim is literally correct and the annotator's own sentence pass agrees), 2 borderline (resp 189, resp 168 - arithmetic verified correct against the table, response marked non-adherent). Estimate **1-3 of 20 finqa non-adherent responses, 5-15%**.
delucionqa: 1 arguable in 12 (resp 77 - "Press on the desired setting option to confirm your choice" vs "press and release the preferred setting option until a check mark appears"; a reasonable annotator could go either way). Estimate **~1 of 12, 8%**.
On the supported side neither subset yielded a single item whose gold looked wrong in 30 items read.

**(b) The response label is measuring a different quantity - large on finqa.**
11 of 20 finqa non-adherent responses (31, 36, 48, 67, 71, 81, 116, 168, 189, 198, 229) have **zero** sentences keyed unsupported by the same annotation pass. Their sentences are all gold-supported; only the whole-response `adherence_score` is 0. Reading them shows why: the response's *final answer* is wrong (wrong year, wrong unit, wrong quantity asked for, self-contradictory), while every sentence is individually grounded. This is answer correctness, not grounding. It is not noise - it is a different label - but a sentence-level grounding model cannot recover it, and **55% of finqa's positive class carries it**. delucionqa: 2 of 12 (106, 130).

**(c) Label absence - large on delucionqa.**
367 of 929 delucionqa scored sentences (39.5%) are unkeyed by the annotator, and on 53 of 172 supported responses (30.8%) the *deciding* (minimum-scoring) sentence is one of them. Those sentences are mostly generator filler with no checkable proposition. Their "supported" status is inherited from the response label, not asserted by anyone. This inflates the apparent false-positive rate without a single incorrect annotation.

**Bottom line**: genuine label noise explains at most ~10% of the errors on either subset. The far larger contributors are a label that measures something else (finqa) and a read that scores something no one labelled (delucionqa).

---

## 5. Verdict per subset

### finqa → **OBJECTIVE problem**, with a DATA component. Not read, not capacity.

**Evidence for objective mismatch**
- 362 of 563 scored sentences (64.3%) assert a number present in no retrieved window; **75.7% of those are gold-supported**. Mean score 0.487 for absent-number sentences against 0.641 for all-numbers-present sentences
- Restricting to gold-supported responses: sentences with an arithmetic marker *and* an absent number score 0.502 (n=230); sentences with neither score 0.687 (n=126)
- 8 of the 15 worst supported items are exactly this: a correct derivation whose conclusion is not in the evidence (147, 205, 112, 101, 50, 184, 195, 148)
- RAGBench-finqa's supported class includes derived values. The training mix (RAGTruth, HaluEval, PsiloQA, VitaminC, TabFact) defines supported as literally entailed. The model is answering the question it was trained on

**Evidence against a read problem** - min-over-sentences (0.6489) beats mean (0.5807) by 0.068 and beats min-over-keyed-sentences (0.6159). The read is the best of the three on finqa.

**Evidence against a capacity problem** - resp 5: 0.9811 and 0.0822 on two lookups into the same 266-char table window, differing only in whether the cell carries a `$` sign. resp 160: 0.9731 on the prose-verbatim figure, 0.0803 on the table-column figure for the adjacent year. The model reads these tables; it applies a literal-presence criterion to them.

**DATA component** - no training source pairs a serialized financial table (`[["row label","value"],...]`) with a claim asserting a *derived* value, nor with a claim whose number differs from the cell only by `$`, comma or unit scaling. Both are learnable surfaces. The register (financial tables) is legal to train on; the RAGBench corpora are not.

**Caveat on the measurement** - 20 unsupported responses, 5 of which carry 57.5% of the loss, and 11 of 20 labelled by a criterion the model is not built for. finqa's AUROC is a low-information, partly off-target number. Any lane claiming a finqa delta under +0.03 is inside the noise the E4 evidence documents.

### delucionqa → **READ problem**, with an OBJECTIVE component. Not data, not capacity.

**Evidence for the read**
- Mean-over-sentences 0.8445 against the shipped min's 0.7975 on identical frozen scores - the aggregation rule costs 0.047
- On 53 of 172 supported responses (30.8%) the min falls on a sentence the annotator never keyed. Worst case resp 164: response score 0.0139, set by "This helps maintain good visibility and air circulation."
- resp 175: the model scores the fabricated-quote sentence 0.6631 - above its own response minimum of 0.2829, which sits on a *gold-supported* sentence. The min rule discarded a correct detection
- Sentence-level AUROC is 0.7767 while response-level is 0.7975; the sentence-level signal is strong (supported 0.885, unsupported 0.643) and the read is not converting it

**Evidence for the objective component** - the two largest false negatives (resp 82 at 21.5%, resp 24 at 16.0% of discordance) are **verbatim quotations of the retrieved evidence** that are gold-unsupported because they answer a different question than the one asked. That is a relevance judgement wearing a grounding label. An entailment scorer will always score a verbatim quotation high; this failure mode is unreachable by any change to the entailment model. Together with resp 65's partial-conjunct fabrication (30.6%), lexical-overlap-shaped items are 68% of delucionqa's loss.

**Evidence against a data problem** - the model handles the register fluently: 0.9944, 0.9660, 0.9636, 0.9223 on verbatim procedural lookups in the same manuals. Its supported-sentence mean is 0.885.

**Evidence against a capacity problem** - the errors are systematic by claim shape (paraphrase, summary, glue, conjunction), not diffuse. Paraphrase blindness is the one genuine model weakness visible here (6 of 15 worst supported items), and it is a training-surface question, not a parameter-count question: score correlates 0.55 with raw content-word overlap.

**Caveat on the measurement** - 12 unsupported responses. Three items are 68.2% of the loss and two of those three are relevance rather than grounding. delucionqa's ±0.10 swings across draws (0.7975 draw 1, 0.8358 draw 2) are one or two items changing rank.

---

## Cross-cutting observations

- **The min read is subset-dependent in the wrong direction.** It helps where responses are short and every sentence is a claim (finqa, 2.25 sentences/response) and hurts where responses are long and padded with glue (delucionqa, 5.05 sentences/response). Any future read proposal must be justified on response structure, not on arena AUROC
- **Verbatim quotation is a hard ceiling.** Three of the top-5 delucionqa false negatives and two of the top-5 finqa false negatives contain only tokens that appear in the evidence. Their unsupportedness lives in binding (which line item, which year, which system) or in scope (does this answer the question), neither of which a sentence-vs-window entailment score represents
- **The two subsets' "quantitative" reputation is one mechanism seen twice.** finqa's derived numbers and delucionqa's partial conjuncts are the same failure: an assertion whose atoms are all present and whose composition is not. This is consistent with H108's quantitative-near-miss lane helping finqa (+0.056 / +0.034) - it targets the composition, not the atoms

## Reproduction

```
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python experiments/grounding-semantic/R14_E4_item_forensics.py
```

Writes `R14_E4_item_forensics.json` (per-subset statistics, discordance ladders) and `R14_E4_worst_items.txt` (verbatim dump of every item cited above). CPU only, ~15 s.
