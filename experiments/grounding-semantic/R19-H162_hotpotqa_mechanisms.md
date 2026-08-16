# R19-H162 Hotpotqa Mechanism Dissection

hotpotqa is multi-hop question answering, where answering requires combining facts from two or more retrieved documents, and it is the flagship's third-lowest arena subset at 2-draw AUROC 0.6706. This memo attributes the residual to named mechanisms measured on the frozen 250-item gate sample, and rules on the brief's structural question: whether the serving read's blindness to cross-window composition is the bottleneck. Analysis-only arm, CPU-only: nothing here trains, no GPU was used, all model scores are read from the banked R19-H161 per-pair dump, and nothing is selected on arena statistics.

**Headline.** The composition need is real and dominant, but the bottleneck is NOT the aggregation architecture. The per-window scorer behaves as a lexical-coverage meter, and on the 71.3% of claim sentences whose support spans documents it produces the same score for true and false claims alike. Three independent cross-window architectures have been built and none helped hotpotqa. The gap is a missing skill with no supply behind it, and it is installable inside the serving shape that already ships.

## Read fidelity and instrument limits

- **Positive control clean** - the max-over-windows read reconstructed from the H161 dump reproduces the banked draw-1 hotpotqa AUROC exactly, 0.6766 vs 0.6766, and the structural fingerprint matches (250 items, 293 sentences, 1,177 pairs)
- **The subset carries 17 negatives** - 233 of 250 items are positive, base rate 0.932; bootstrap 95% CI on the draw-1 read is [0.5683, 0.7796], width 0.211
- **Every absolute hotpotqa number in the campaign is a 17-negative reading** - the 0.6706 flagship value, the 0.7039 record and the 0.6514 early reads all sit inside one another's confidence intervals; only PAIRED comparisons on the same 250 items (same-seed deltas, ablation twins) carry usable resolution
- **The sample is effectively single-sentence** - 1.172 sentences per item, so the min-over-sentences axis barely operates and hotpotqa is a near-pure test of the sentence-versus-window scorer
- **Geometry is uniform** - 3.99 documents per item, 4.02 windows per sentence with SD 0.19; window-count dispersion is almost absent, so count-aware corrections are per-subset constants here
- **Artifacts** - `R19-H162_hotpotqa_probe.py` and `R19-H162_hotpotqa_families.py` (scripts), `R19-H162_hotpotqa_mechanisms.json` and `R19-H162_hotpotqa_families.json` (measurements), `R19-H162_hotpotqa_sentences.parquet` and `R19-H162_hotpotqa_families.parquet` (per-sentence records), `R19-H162_hotpotqa_eyeball.md` and `R19-H162_hotpotqa_families_eyeball.md` (read items), logs `logs/R19-H162_hotpotqa_probe.log` and `logs/R19-H162_hotpotqa_families.log`

## The multi-hop share, measured

The brief asks what fraction of claim sentences genuinely require two documents. Measured by greedy set cover over each sentence's lexical anchors against the item's document list.

- **209 of 293 claim sentences, 71.33%, need two or more documents** to cover their matched anchors; 84 sentences, 28.67%, are covered by a single document
- **Documents needed** - 1 doc 84 sentences, 2 docs 174, 3 docs 34, 4 docs 1
- **Independent method agreement 98.29%** - the R16-H140 G0 census flagged 71.67% cross-window at the anchor-span level using a different construction; this census flags 71.33% cross-document by set cover, and the two agree sentence by sentence
- **The single-document escape hatch is small** - only 31 of the 209 multi-document sentences, 14.8%, have one document carrying 80% or more of the anchors; 53 have their best document below 60%
- **Mean best-single-document anchor containment is 0.7283, rising to 0.8647 when a second document is allowed** - the second document adds 0.1363 of coverage on average
- **Caveat inherited from the H140 census** - lexical anchor co-occurrence is a proxy and errs both ways; it overcounts when a single window entails a claim without carrying every surface form, and undercounts paraphrased support. It bounds where cross-document information could live, not how much is needed

## The failure: partial-support saturation

The decisive measurement. Split the 293 sentences by hop class and read the flagship's max-window logit.

- **On single-document sentences the model separates cleanly** - positive mean -0.308 vs negative mean -2.523, gap +2.216, sentence-level AUROC 0.7286
- **On multi-document sentences the separation is ZERO** - positive mean -3.6646 vs negative mean -3.6640, gap -0.0006, sentence-level AUROC 0.6017
- **Item-level, the same split** - items whose sentences are all single-document read AUROC 0.8922 (n 55, 4 negatives); items with any multi-document sentence read 0.6691 (n 195, 13 negatives)
- **158 of 192 multi-document positive sentences, 82.3%, score below the MEAN of single-document NEGATIVES** - a true composed claim is scored lower than a false single-document claim four times in five
- **The score tracks coverage, not truth** - correlation of the max logit with best-single-document containment is +0.5596, with the gain from a second document -0.4243, with the argmax window's token containment +0.6082
- **The coverage penalty survives its confounds** - partial correlation of the max logit with documents-needed, controlling for anchor count, is -0.5138 against a raw -0.5417, while anchor count alone reads only -0.2175; in every anchor-count bin the multi-document mean sits 2.5 to 4.5 logits below the single-document mean
- **The label collapse survives its confounds too, at reduced size** - negatives are shorter than positives here (86.7 vs 110.4 chars), so the raw zero gap is length-flattered; regressing the max logit on label with sentence length and anchor count as covariates gives a label coefficient of **+0.251 on multi-document sentences against +1.766 on single-document**, a 7-fold collapse, and a length-matched band of 70 to 115 chars reads a gap of -0.170

## Claim families

Multi-document sentences split into two structurally distinct families, and the lever differs by family. Classification is rule-based (conjunction and comparative markers, clause-level document assignment, elided-token detection) and the exemplars were read.

- **bridge_entity, 104 sentences, 35.49% of all, the largest family** - the claim names two endpoints and elides the intermediate entity that links them, so no window carries both endpoints. Exemplar: "The title of the memoir written by the honoree of the Black and White Ball is 'Personal History'", where one document names the honoree and another attributes the memoir. Mean 2.13 shared-but-elided content tokens between the covering documents
- **single_hop, 84 sentences, 28.67%** - one document covers the claim
- **conjoin_attrs, 56 sentences, 19.11%** - one attribute asserted about each of two entities, each entity's attribute in a different document, joined by a conjunction or a comparative. Exemplars: "No, Guillermo Cabrera Infante was Cuban, while Guillaume Apollinaire was French"; "Panicum has the higher number of species, with about 450 species, compared to Populus, which has 25-35 species". Mean minimum clause containment 0.7214, so each clause IS well covered, just by different documents
- **multi_doc_other, 44 sentences, 15.02%** - cross-document with no detected conjunction or elided bridge; no single mechanism
- **Discrimination by family** - bridge_entity positive mean -3.680 vs negative -3.110, gap **-0.570 with the WRONG SIGN**, sentence AUROC 0.5574; conjoin_attrs gap +0.379 with bootstrap CI [0.127, 0.640], AUROC 0.6635; single_hop gap +2.216, AUROC 0.7286
- **Negative mass concentrates on bridges** - of the 23 negative sentences, 10 are bridge_entity, 6 single_hop, 4 conjoin_attrs, 2 multi_doc_other, 1 conjoin_attrs_same_doc
- **Evidence selection degrades with structure** - the argmax window lands on the highest-containment document 86.9% of the time on single_hop, 59.6% on bridge_entity and **48.2% on conjoin_attrs**, against 25% chance over four documents; conjoin_attrs is the worst-selected family in the subset
- **Negatives are false compositions, not off-topic text** - read exemplars: "Yes, Ainslee's Magazine and The Australian Women's Weekly are both monthly magazines" (false conjunct, scored -4.85), "Candleshoe came before The Strongest Man in the World" (false cross-document ordering, -4.63), "The German rally driver drove the Lancia Delta Group A" (broken bridge, -4.36). These sit in the same -4.1 to -5.7 band as TRUE cross-document claims

## Why the cross-window readout made hotpotqa worse

The R16-H140 arm replaced the hard max with a learned attention readout over window embeddings, precisely to enable composition, and hotpotqa was the worst mover at -0.0518, seed-replicated at -0.0427 with same sign 4 of 4. The account has three legs, and the third is the load-bearing one.

- **The aggregation axis has the LEAST purchase on hotpotqa of any subset** - recomputing the arena under mean pooling over the same banked per-window logits, hotpotqa moves **-0.0023**, the smallest magnitude of the ten; techqa moves -0.1403, expertqa -0.0991, tatqa -0.0908, finqa +0.0589. Fixed soft poolings on hotpotqa read 0.6743 (mean) to 0.6981 (softmax t=4) against max 0.6766, all inside the subset's own 0.211-wide CI
- **Nothing can be composed out of signal-free inputs** - the per-window logits carry a label gap of -0.0006 on exactly the multi-document sentences the readout was built to serve, so any reweighting of those windows moves rank by noise alone; hotpotqa, having the least per-window signal, receives the most pure variance from a learned reweighting
- **The clean single-variable test of cross-window conditioning moved hotpotqa by +0.0028, a null** - R16-H142 G1 was an init-fingerprint-paired ablation (identical init 9d679fcb, matched permutation) whose ONLY difference was activating a zero-init adapter conditioned on the mean-pooled window-set context. Its arena mean fell -0.0323 and pubmedqa -0.1113, but hotpotqa did not move. The channel that could compose across windows was switched on under the strictest control the campaign has run, and the motivating subset was indifferent to it
- **Consequently the H140 loss is re-ranking, not failed composition** - the readout was trained on a public slice (RAGTruth plus manufactured lanes, mean window-set size 2.64) whose support is single-window; its learned mapping shifted score distributions between registers, banking pubmedqa +0.0711 and emanual +0.0400 while pushing hotpotqa and tatqa down. The H141 capacity-matched control named the same failure shape, window-count out-of-distribution extrapolation, for the scalar family
- **A third arm agrees** - R18-H156 trained a learned aggregator over per-window logits through the trunk; its serving-read swap was neutral at -0.00008 and hotpotqa read -0.0466, part of a -0.0250 whole-model regression traced to gradient dilution of the MIL selection pressure

## The architecture is not blind, it is untrained

The brief's premise, that the architecture cannot compose by construction, needs one amendment that changes the verdict.

- **A cross-window channel already exists in the shipped serving shape** - `pair_logits(cls, ctx)` in `R16-H142_G1_arm.py` computes `task_head(cls) + adapter([LN(cls); LN(ctx)])`, where `ctx` is the mean-pooled CLS over the entire window set for that sentence, so every window's logit CAN see a summary of every other window
- **In the banked flagship that channel is inert** - `models/R18-H150-arm-draw1/adapter.pt` carries `adapter_active = False` and `adapter.2.weight` and `adapter.2.bias` are exactly zero, so the flagship's logit reduces to `task_head(cls)` and depends only on the sentence-window pair. The blindness is real at serve, but it is a trained-off channel, not a missing one
- **The supply has never presented a composed positive** - the R16-H142 executor census found 100% of the incumbent's 685,670 training rows had a size-1 window ensemble under the original 1,500-char truncation; after amendment A1's untruncated windowed presentation the mix reads mean 1.507 windows per row with 20.1% multi-window, and vitaminc at 54% of the mix trains at exactly 1.00. No lane in the mix contains a positive whose support requires more than one evidence document
- **The MIL max objective already defines the correct target for a composed positive** - for a positive bag, some window must score high, so the skill "score this window high because it supplies its share and the remainder is present in the bag" is learnable without any architecture change. The model currently scores a window on how much of the claim that window covers, which is a different function

**Verdict: MISSING_SKILL, with an architectural amplifier that is already closed.** The composition need is dominant (71.3%) and the model's discrimination on it is at chance, but three independent architectural interventions have moved hotpotqa by -0.052, +0.003 and -0.047, and the aggregation axis has less purchase on this subset than on any other. What is absent is training supply that forces the pair scorer to treat a partially-covering window as evidence of a composed claim.

## Mechanisms

Each mechanism is named in the register of "bind a value to its column header", with the measurement that makes it a bottleneck, a held-out probe with its chance level, and a lane candidate.

### 1. bridge_entity

- **Definition** - bind the claim's two named endpoints through an intermediate entity that the claim omits and that appears in both covering documents
- **Bottleneck evidence** - 104 of 293 sentences (35.49%), the largest family; positive mean -3.680 vs negative mean -3.110, gap **-0.570, sign inverted**; sentence AUROC 0.5574 against single_hop's 0.7286; carries 10 of the 23 negative sentences; argmax window lands on the highest-containment document 59.6% of the time against single_hop's 86.9%
- **Probe design** - 1,000 held-out synthetic items from a two-relation join. Document A states `R1(E) = M`, document B states `R2(M) = v`, plus two distractor documents. Positive claim elides M: "the R1 of E has R2 v". Negative takes v from a distractor entity M' present in a third document, so every surface token of the claim still appears somewhere in the evidence bag and only the chain is broken. Entity-disjoint from any training lane. **Chance 0.500 AUROC**, and the flagship's current standing on the natural analogue is 0.5574
- **Lane candidate** - rule-based generator over already-banked TabFact tables (CC-BY-4.0, 16k Wikipedia tables): pick two tables sharing a key column, emit each row as its own document, and template the claim with the join key elided. The generator rule is the `quant_misbind` precedent applied to a join instead of a cell
- **Contamination** - CLEAR. TabFact is banked, licence-verified, and is not a RAGBench source or derivative. The generator never touches HotpotQA. HoVer is BLOCKED and already recorded in this project's survey as "HotpotQA-derived - walled"
- **Already covered by** - null. No banked corpus contains claims whose support requires two separate evidence documents. MiniCheck (MIT) is the nearest, multi-fact and multi-sentence, but its support is assembled within ONE document

### 2. conjoin_attrs

- **Definition** - verify a claim that asserts one attribute about each of two entities, where each entity's attribute lives in a different document, so support requires both conjuncts
- **Bottleneck evidence** - 56 of 293 sentences (19.11%); positive mean -4.259 vs negative mean -4.638, gap +0.379 with bootstrap CI [0.127, 0.640] against single_hop's +2.216, a 6-fold collapse; sentence AUROC 0.6635; the argmax window lands on the highest-containment document **48.2%** of the time, the worst of any family and barely double the 25% chance over four documents; mean minimum clause containment 0.7214 shows each conjunct IS well covered, just in different documents
- **Probe design** - 1,000 held-out synthetic two-entity conjunction items, each entity's attribute in its own document plus two distractors. Positives assert both conjuncts truly. Negatives flip exactly ONE conjunct and leave the other true, so a coverage meter cannot separate them. A comparative sub-leg ("X has more A than Y") reuses the relational-compare construction across documents. **Chance 0.500 AUROC**
- **Lane candidate** - same generator, different template: two rows from different banked tables emitted as separate documents, conjunction and comparative claim templates over them, negatives by single-conjunct value swap. TabFact CC-BY-4.0 and the banked FEVEROUS slice already in the H108 quant lane family supply the rows
- **Contamination** - CLEAR, same reasoning as bridge_entity; rule-generated from banked tables with no HotpotQA lineage at any remove
- **Already covered by** - PARTIALLY. The comparative leg is the relational-compare skill the probe bank already reads at 0.51, at chance, never installed; this mechanism is that same missing skill compounded with a cross-document evidence split. The conjunction leg is covered by nothing banked

### 3. partial_support_credit

- **Definition** - score a window that supplies only its share of a claim as evidence FOR that claim when the remainder is present elsewhere in the window bag, and against it when the remainder is absent or contradicted
- **Bottleneck evidence** - this is the cross-cutting mechanism behind both families above. On the 209 multi-document sentences the label gap is -0.0006 raw and the length-adjusted label coefficient is +0.251 against single-document's +1.766; 82.3% of multi-document positives score below the mean of single-document negatives; the max logit correlates +0.5596 with best-single-document containment and -0.4243 with the coverage a second document adds, so the scorer is measuring coverage rather than support
- **Probe design** - a paired contrast rather than a new item type: for each held-out composed item, score the full composed claim and its two decomposed halves against the same bag. A model with the skill scores the composed positive near its halves; a coverage meter scores it near the floor. Reported as the composed-minus-decomposed score deficit, **chance value 0.0 logits**, flagship expectation strongly negative
- **Lane candidate** - not a separate lane. It is the presentation constraint that the two generators above must satisfy: every composed positive must be presented as a MULTI-DOCUMENT bag under the 1,500/750 windowed MIL objective, so that no single window fully supports and the max objective is forced to reward the best partial window
- **Contamination** - CLEAR, generator-only
- **Already covered by** - null, and the supply census explains why: 100% of the pre-A1 training rows were single-window, the post-A1 mix is mean 1.507 windows per row, and vitaminc at 54% of the mix is exactly 1.00

## Build first

**bridge_entity**, using a generator that yields conjoin_attrs from the same machinery.

- **Largest and worst** - 35.49% of claim sentences and the only family whose score gap has the wrong sign, at AUROC 0.5574 against a 0.5 floor; it also carries the largest share of the scarce negative class, 10 of 23
- **One builder serves two mechanisms** - a two-table join over banked TabFact rows produces bridge items by eliding the join key and conjunction items by emitting two independent rows, so the marginal cost of the second mechanism is a template
- **No new architecture** - the lane trains inside the shipping serving shape under the existing MIL max objective; the three architectural routes are closed by H140, H142 G1 and H156
- **Mandatory paired construction, stated as a risk** - a lane that teaches "credit a partially-covering window" WITHOUT composed negatives will install over-crediting, which is the dominant finqa failure the H157 autopsy named. Every composed positive must ship with a composed negative whose missing conjunct or broken bridge is absent from the bag, or the lane will damage the tabular and financial subsets

**Expected arena movement**: hotpotqa +0.02 to +0.06 if the skill installs, which is +0.002 to +0.006 on the ten-subset arena mean against the 0.71549 flagship. The upper end assumes multi-document sentences reach the single-document family's 0.7286 discrimination. Confidence is LOW and the honest downside is real: three prior arms aimed at this subset read -0.052, +0.003 and -0.047, the subset's own 95% CI is 0.211 wide on 17 negatives, and a lane that installs partial-support credit without matched negatives is more likely to cost finqa and tatqa than to buy hotpotqa. Any build must be a paired-draw arm with the tabular subsets as pre-registered guardrails.

## What was ruled out

- **Window geometry** - 0 of 3,556 hotpotqa evidence sentences are cut by every window and the anchor-span histogram places 0 sentences in the 1,500 to 6,000 char band, so the dispersion is purely cross-document; the H140 census already closed the neighbour-channel and SAT-aligned-windowing branches on this evidence
- **Aggregation form** - hotpotqa is the least aggregation-sensitive subset of the ten at -0.0023 under mean pooling, and every fixed soft pooling sits inside the subset's CI
- **Learned cross-window conditioning** - the H142 G1 init-paired ablation moved hotpotqa +0.0028
- **Sentence length as the explanation** - the coverage penalty survives partial correlation on anchor count at -0.5138, and the label collapse survives length-and-anchor adjustment at a 7-fold reduction
- **The single-hop restatement hypothesis** - only 14.8% of multi-document sentences have one document carrying 80% or more of their anchors, so the residual cannot be re-read as mostly single-hop restatements
