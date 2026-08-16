# R19-H162 Pubmedqa Mechanism Dissection

`pubmedqa` (biomedical question answering, the arena's lowest subset at 2-draw AUROC 0.6096) fails for a different reason than `finqa` does: there is no single at-chance skill like relational derivation. The per-sentence verifier works, reading 0.6700 against RAGBench's own sentence-level support annotations, but it adds only +0.0040 over a bag-of-words containment ratio and the fixed min-over-sentences aggregation then destroys 0.0807 of it - the largest such loss in the arena. Analysis-only arm: nothing here trains, tunes or selects on arena statistics; every lever named below builds from public data and validates off-arena.

- **Read fidelity** - the analysis rides the banked R19-H161 per-pair dump; its own control reproduces the banked windowed pubmedqa AUROC bit-exact (h150d1 read 0.5893 vs banked 0.5893, delta -0.00001; h150d2 0.6298 vs 0.6298, delta -0.00004). Structural fingerprint 250 items / 1,314 sentences / 6,570 pairs
- **Artifacts** - corpus census `R19-H162_pubmedqa_explore.py` + `_explore.json` + `_sents.parquet`; failure analysis `R19-H162_pubmedqa_analyze.py` + `_analysis.json` + `_sent_h150d1.parquet`; sentence-truth diagnosis `R19-H162_pubmedqa_sentlabel.py` + `_sentlabel.json` + `_sentlabel.parquet`
- **Headroom** - faithful-oracle ceiling 0.7789 under the shipped read (`R12_label_ceiling_result.json`), flagship 0.6096, gap **0.169** - the largest per-subset headroom in the arena
- **Sample** - 250 items, 77 negatives, 1,314 sentences of which 1,301 align to a RAGBench annotated sentence at similarity >= 0.6 (align rate 0.9901, mean similarity 0.9842); 165 sentences are annotated unsupported. Per-class counts below carry binomial SEs and the thin ones are labelled

## Corpus shape

The evidence is exactly five PubMed abstract fragments per item - typically the target study's objective plus four topically adjacent abstracts - and a third of them state only what a study set out to do. Only 1.1% of the 1,250 documents end mid-sentence, so hard truncation is rare; the missing results are missing because the retrieved fragment is an objective, not because it was cut.

- **Every document fits in one window** - mean 372.5 chars, max 1,354, 0.0% over the 1,500-char window; exactly 5.00 windows per sentence with zero dispersion, the shallowest geometry in the arena (techqa 22.49, expertqa 10.71). Window boundary is not a live loss source here and count-aware aggregation is a per-subset constant, as R16-H141 already measured
- **Responses are long** - 5.256 sentences per item (mean), median 5, max 12; only 5 single-sentence items. Among the arena's weak subsets this is the highest sentence count (finqa 2.25, hagrid 2.15, hotpotqa 1.17)
- **Unsupported sentences are sparse** - 12.68% of aligned sentences, so a negative item carries about 0.67 bad sentences among 5.26. expertqa and techqa have more sentences but 34% unsupported, so their negatives carry 2.2-2.3 bad sentences each
- **The evidence is mostly aims, not results** - 33.7% of the 1,250 evidence documents carry an explicit objective or aim statement, only 13.0% carry any result language (p-values, significance, comparisons, "found that")
- **Hedging is pervasive and not a label signal** - 32.7% of sentences hedge; hedged sentences are 22.8% unsupported against 7.9% for unhedged, but hedging saturates both classes and 29.1% of sentences carry an explicit inference marker
- **Numerals are secondary** - 27.9% of sentences carry a number, 7.8% carry one absent from the evidence; this is a prose subset, not a tabular one

## The headline diagnosis

The verifier is not broken at the sentence level; it is a lexical-overlap detector wearing a semantic model, and the aggregation then eats most of what it has.

- **True sentence-level AUROC 0.6700** (h150d1, supported vs unsupported sentence, RAGBench annotation as truth) against a bag-of-words containment ratio at **0.6660** - the model's semantic margin over word counting is **+0.0040**, second-worst in the arena after hagrid at -0.0977 (tatqa +0.1237, hotpotqa +0.1204, covidqa +0.0872, delucionqa +0.0497, emanual +0.0469, expertqa +0.0186, techqa +0.0090, finqa +0.0091)
- **Aggregation dilution +0.0807** - sentence AUROC 0.6700 collapses to item AUROC 0.5893 under min-over-sentences. This is the only large positive dilution in the arena; six subsets gain from aggregation (tatqa -0.1134, finqa -0.1147, hagrid -0.1013, delucionqa -0.0235, emanual -0.0159, hotpotqa -0.0144, expertqa -0.0126) and the two that lose lose little (covidqa +0.0260, techqa +0.0097)
- **Why min is worst here** - a negative item must have its one bad sentence found among 5.26; the argmin lands on an annotated-unsupported sentence 52.6% of the time against a 35.0% within-negative base rate (lift 1.51x). Meanwhile a positive item takes the minimum of 5-6 supported scores, and that minimum reads **-3.933** on average, BELOW the mean unsupported-sentence score of **-3.159**. The left tail of the supported distribution, not the unsupported sentence, sets the item score on positives
- **Only subset where a trivial heuristic beats the model** - ranking items by fewer response sentences alone reads 0.6347 against the model's 0.5893. Every other subset's model wins, most by wide margins (covidqa 0.7685 vs 0.4865, delucionqa 0.8009 vs 0.4249, tatqa 0.7842 vs 0.6416). Diagnostic only - length is not a lever and selecting on it would be selection on the benchmark
- **Error split is exactly balanced** - at the in-sample macro-F1-optimal threshold (the R17-H147 stated choice, nothing tuned on it) h150d1 makes 88 errors: 44 false positives (0.571 of 77 negatives, binomial SE 0.056) and 44 false negatives (0.254 of 173 positives, SE 0.033). Both directions need fixing; a negative-only lane addresses half the loss

## Mechanism taxonomy

The taxonomy is grounded in RAGBench's own free-text explanation of why each of the 165 unsupported sentences failed, classified by a rule layer with the family regexes recorded in `R19-H162_pubmedqa_sentlabel.py`. Families genuinely co-occur, so both a multilabel count (any match) and a primary-class count (first match, the regexes are ordered specific-first) are quoted. **The AUROC columns are computed on the PRIMARY-class members only**, so read them against the primary n, not the multilabel n. A further 41 explanations (24.8%) fall to `unclassified` under primary assignment and read model 0.6582 vs lexical 0.6076.

| family | multilabel n | primary n | primary share (SE) | model AUROC vs supported | lexical AUROC | model margin |
|---|---|---|---|---|---|---|
| inference_not_stated | 83 | 61 | 0.370 (0.038) | 0.6861 | 0.7074 | **-0.0213** |
| aim_vs_finding | 26 | 26 | 0.158 (0.028) | 0.6463 | 0.6641 | **-0.0178** |
| scope_overextension | 17 | 10 | 0.061 (0.019) | 0.6832 | 0.6860 | -0.0028 |
| relation_not_attested | 15 | 12 | 0.073 (0.020) | 0.6338 | 0.7144 | **-0.0806** |
| contradiction | 10 | 10 | 0.061 (0.019) | 0.6833 | 0.5370 | **+0.1463** |
| entity_substitution | 5 | 4 | 0.024 (0.012) | n<5, not read | - | - |
| false_absence | 2 | 1 | 0.006 (0.006) | n<5, not read | - | - |

- **The model beats word counting on exactly one family** - contradiction, +0.1463, and contradiction is 6.1% of the failures. On the two largest families it is BELOW a containment ratio. Read plainly: the only semantics the verifier has installed for this register is negation and polarity, which is what VitaminC and the near-miss lanes teach
- **No family is at chance** - every readable family sits 0.63-0.69, so unlike finqa's derivation there is no single dead skill; the deficit is broad and shallow
- **Inference marking is the hardest slice** - the 29.1% of sentences carrying an explicit inference marker ("suggests that", "indicating that", "therefore", "implies") read model 0.6028 vs lexical 0.6157, the model's worst slice. Within them, 291 sentences are annotated supported and 87 unsupported at mean scores -2.938 and -3.431 - the two classes are lexically and structurally identical and the model cannot split them
- **Meta-commentary is handled, not broken** - the 30.1% of sentences that talk about the evidence rather than asserting biology read model 0.7179 vs lexical 0.5911, the model's best slice. On positive items, sinking sentences are LESS likely to be meta than non-sinking ones (0.215 vs 0.321). This independently re-confirms the R12 precursor P-B kill: the discourse-marker-sinks-the-min story is false, and its oracle bound was +0.0065

## What the annotator's explanations actually say

Verbatim exemplars, one per family, from `sentence_support_information[].explanation`.

- **inference_not_stated** - "This inference, although logical based on the setup provided in documents related to the research on NGAL, is unsupported specifically by the documents"; "It is a plausible synthesis based on the provided differences in orientation, but it does not directly cite these complementarities"
- **aim_vs_finding** - "The hypothesis is mentioned in Document 0 but no supporting evidence or results are provided in any of the documents"; "suggesting anticipation of results without results presented results in partial support"; "goes beyond documented findings as no specific results are cited"
- **relation_not_attested** - "Documents 2 and 3 do discuss patient characteristics and glycemic control improvement but do not link these to patient readiness to change"; "describes influences of sociohistorical values on medical education but does not specifically connect to pharmacy practices"
- **scope_overextension** - "the documents provide weight data but do not conclusively show that oral-only is associated with 'better' weight gain compared to other methods. In fact, the tube-only method showed better outcomes in z-scores"
- **entity_substitution** - "misattributes the aim from Document 0 (which is actually about total hip replacement, a different procedure) to be about hip arthroscopy"; "the question is about the retinomax, not the PlusoptiX S04"

## The false-negative half

44 of the 88 errors are supported responses scored too low, and they have a single visible signature.

- **The sinking sentence of a positive item is its lowest-lexical-overlap sentence** - mean containment 0.344 against 0.536 for non-sinking sentences of the same items, at mean score -3.933
- **Those sentences are genuinely supported** - by construction every sentence of a positive item is annotated supported; the model is penalising abstraction, not detecting a real defect
- **Exemplars** - "The overall rate of axillary lymph node involvement across the studies was variable, ranging from 19.0% to 38.3% of patients" (containment 0.07, score -4.96, a range aggregated across several abstracts); "En-bloc resection resulted in higher survival rates, indicating that a more comprehensive surgical approach is recommended" (containment 0.06, score -4.88); "This suggests that with proper maintenance and calibration, aneroid sphygmomanometers can be accurate in hospital and clinic settings" (containment 0.09, score -4.82)
- **The training mix teaches the opposite reflex** - every near-miss lane banked in this campaign builds negatives as HIGH-overlap minimal edits by its own card's description (VitaminC "differs from its positive by one number, entity or qualifier and nothing else"; LettuceDetect "localized replacement edits"; quant_misbind "zero pairs differ in anything but the numeral"). Nothing in the mix teaches that LOW overlap can still be support
- **Honest limit** - some of these sentences are the same SHAPE as the unsupported inference class and differ only in how far the annotator let the inference run. Part of this residual is annotator threshold, not model deficit, and no measurement here separates the two

## The ceiling is a separate, structural quantity

- **The whole 0.208 ceiling loss on pubmedqa is cross-document conjunctive support** - the faithful oracle reads 1.0 with perfect labels, 0.987 after the H92 splitter, 0.987 after the document cap, 0.987 under a lenient cross-window read, and **0.7789** only when support is required to fit in ONE window (`R12_label_ceiling_result.json`)
- **Window equals document here** - since no pubmedqa document exceeds 1,500 chars, "support must fit one window" means "support must come from one abstract". 15.6% of pubmedqa's 935 annotated supported sentences draw on more than one document, and **42.8% of positive items carry at least one** such sentence against 26.0% of negative items
- **Not a lane target** - max-over-windows is an OR and read amendments are closed (R12: hard_min 0.7355 against a best alternative 0.7230; the aggregator line is measured shut). This is recorded so the 0.169 gap is not mistaken for 0.169 of trainable skill
- **Bars stay ceiling-blind** per the standing R12 ruling 6; the ceiling informs prioritisation only

## Mechanisms, probes and lanes

Each mechanism is stated in the register of "bind a numeric value to its column header". Probes follow the R14-H133 convention: held-out minimal pairs, source-disjoint from every train split, zero arena, zero gold, read on both the arm and a banked control checkpoint; chance is AUROC 0.5.

### 1. assert_vs_infer - BUILD FIRST

- **Definition** - separate a proposition the evidence STATES from one the evidence merely makes PLAUSIBLE
- **Bottleneck evidence** - 50.3% of the 165 annotated unsupported sentences (n=83, largest family); model 0.6861 vs lexical containment 0.7074, i.e. **0.0213 below word counting**. On the 29.1% of sentences carrying an explicit inference marker the model reads 0.6028 against lexical 0.6157, its worst slice, and the 291 supported / 87 unsupported inference-marked sentences are not separated at all
- **Probe design** - rationale-deletion minimal pairs over SciFact's SUPPORT rows, which ship expert rationale sentence keys. For each claim build (a) the intact abstract, (b) the same abstract with ONLY the rationale sentence removed and every other sentence kept. (a) is supported, (b) is topically identical but no longer states the claim. AUROC(a vs b), balanced, chance 0.5. Roughly 508 SUPPORT rows are available; held-out at the abstract level and disjoint from any lane build
- **Lane candidate** - the same deletion operator run as a generator over the already-banked, already-gated corpora that carry rationale or span annotations: SciFact (rationale sentence keys), FAVA (typed spans, including its own `unverifiable` and `relation` span types, 30,073 rows CC-BY-4.0), MiniCheck C2D/D2C (14,395 rows MIT, multi-sentence assembly by construction). Deterministic, no generation pass, no new acquisition
- **Contamination** - CLEAR. All three ran the R14-H136 8-gram Jaccard instrument against the ten walled arena corpora and passed: SciFact max_fraction 0.0 (`R13-scifact_gates_result.json`), MiniCheck 0.0, FAVA 0.000116. None is PubMedQA-derived
- **Already covered by** - partially. The corpora are banked as SUPPLY ONLY; no lane teaches deletion-of-the-stating-sentence, and no banked lane's negatives are constructed by REMOVING evidence rather than corrupting the claim
- **Licence caveat** - SciFact carries a recorded discrepancy: upstream AI2 states CC BY 4.0 (claims) + ODC-By (abstracts), the HuggingFace mirror tags cc-by-nc-2.0. The author's 2026-08-09 ruling treats upstream as authoritative; a non-commercial reading would bar a shipped model trained on it. MiniCheck (MIT) and FAVA (CC-BY-4.0) carry no such issue and can supply the lane alone

### 2. paraphrase_support

- **Definition** - recognise that a claim is supported when it RESTATES the evidence in different words, rather than requiring shared tokens
- **Bottleneck evidence** - carries the false-negative half of the error budget (44 of 88 errors on h150d1, 0.254 of positives, SE 0.033). On positive items the sinking sentence has mean containment 0.344 against 0.536 for its non-sinking siblings and scores -3.933, below the mean unsupported sentence at -3.159. Because the item score is that minimum, these false lows set the score on 173 of 250 items
- **Probe design** - supported (claim, evidence) pairs stratified by token containment into deciles; the probe statistic is the model's mean score in the lowest containment decile against the highest, and AUROC of supported-low-overlap against unsupported-high-overlap. A model with the skill shows a flat score profile across deciles; the flagship's profile is the measurement. Chance for the AUROC leg is 0.5
- **Lane candidate** - FActScore (549 labelled biographies, order-10k human atomic-fact S/NS judgments, MIT; atomic facts are abstractive restatements of a full Wikipedia article by construction) and AttributionBench (16,524 rows Apache-2.0 after the ExpertQA/HAGRID carve-out; generative-search answers are heavily paraphrased). Select the supported rows in the low-containment tail as a positive-side lane - the mirror of every existing near-miss lane
- **Contamination** - CLEAR. FActScore gate PASS, AttributionBench gate PASS with the ExpertQA/HAGRID rows dropped by construction and the gate re-deriving zero walled rows
- **Already covered by** - no. Every banked near-miss lane is negative-side and high-overlap by its own card's description; this is the untaught direction
- **Honest limit** - this is the least crisply atomic of the five. "Support survives paraphrase" is close to "be a better entailment model", and the probe measures a score profile rather than a discrete skill

### 3. aim_vs_finding

- **Definition** - bind a proposition to its evidential status: a study's stated OBJECTIVE or HYPOTHESIS is not its RESULT
- **Bottleneck evidence** - 15.8% of unsupported sentences (n=26, SE 0.028); model 0.6463 vs lexical 0.6641, again below word counting. Structurally supported by the corpus census: 33.7% of the 1,250 evidence documents carry an explicit aim statement while only 13.0% carry any result language, so the most common thing the evidence offers is precisely the thing a claim must not be credited against
- **Probe design** - minimal pairs sharing every content word and differing only in evidential status. Evidence (a) is a trial's stated primary outcome measure ("To assess whether X affects Y"), evidence (b) is the same trial's posted result for that measure; the claim is the result ("X did not significantly affect Y"). (b) supports it, (a) does not. AUROC(b vs a), balanced, chance 0.5. Lexical overlap between claim and both evidences is matched by construction, so a containment baseline reads chance and the probe isolates the skill
- **Lane candidate** - ClinicalTrials.gov registrations, rule-based. Completed interventional trials with posted results carry both the pre-stated primary outcome measure and the posted result for it as separate structured fields, which is the minimal pair for free at scale (order 10^4-10^5 trials with posted results). Licence: US National Library of Medicine, works of the US Government, public domain; the CTTI AACT redistribution is freely available. Licence must be re-verified at pull time per the directory's standing convention
- **Contamination** - CLEAR by construction: PubMedQA is built from PubMed abstracts, ClinicalTrials.gov registrations are a separate corpus. The R14-H136 8-gram gate runs regardless and is the binding check
- **Already covered by** - no. No banked corpus separates a study's aim from its result; SciFact's NEI class (485 rows) is the nearest analogue but is not aim-versus-result by construction
- **Build cost** - the highest of the three leaders: new acquisition, a dataset card, a fetch script entry and a gate run before any lane build

### 4. relation_attested

- **Definition** - bind a relation to the sentence that ASSERTS it, not to the co-presence of its two arguments in the evidence
- **Bottleneck evidence** - the largest model-versus-lexical deficit of any family, model 0.6338 against lexical 0.7144, **-0.0806**; but only 9.1% of unsupported sentences (n=15) and at that count the deficit itself carries a wide interval
- **Probe design** - triples over any corpus with sentence-level rationales: evidence containing sentence A (about entity X) and sentence B (about entity Y) but no sentence linking them; claim (a) restates A, claim (b) restates B, claim (c) asserts the X-Y relation. AUROC(a,b vs c), chance 0.5. The construction is exactly the "conflate two attested facts" negative
- **Lane candidate** - MiniCheck C2D/D2C (MIT, 14,395 rows, built so support must be assembled from several sentences) and VitaminC (~489k contrastive Wikipedia revisions, CC-BY-SA-3.0). The relation-conflation negative is a deterministic recombination over multi-sentence evidence, no generation pass needed
- **Contamination** - CLEAR. MiniCheck gate PASS at max_fraction 0.0; VitaminC is banked and Wikipedia-derived
- **Already covered by** - partially, and with a recorded warning. MiniCheck is banked as supply for multi-sentence assembly, but no lane builds relation-conflation negatives. The warning is the R11 precedent: adding Wikipedia-register supervision drove pubmedqa from 0.5665 to **0.4783, below chance**, and the VitaminC negative-class fix recovered it only to 0.5466. Wikipedia-register mass is measured harmful to this subset
- **Structural cap** - the read cannot see cross-document relations at all (max over five single-window abstracts), and the ceiling arithmetic puts the whole 0.208 ceiling loss there. A lane can teach within-document relation attestation; it cannot recover the cross-document half

### 5. scope_bind - UNDER-EVIDENCED, recorded not recommended

- **Definition** - bind a finding to the subgroup, condition and comparator it was measured on
- **Bottleneck evidence** - 10.3% multilabel (n=17), model 0.6832 against lexical 0.6860, a margin of -0.0028 that is indistinguishable from zero at this count. The exemplars are real and clean ("group 1 versus group 2 showed no significant difference" widened to "rejection and delayed graft function"), but n=17 does not support a build decision
- **Probe design** - subgroup-swap minimal pairs: evidence states a finding for subgroup S, claim (a) states it for S, claim (b) states it for a sibling subgroup named in the same evidence, claim (c) states it universally. Chance 0.5
- **Lane candidate** - the quant_misbind generator family extended from table cells to subgroup labels in prose; the machinery exists (`R17-H146_lane.parquet`, 30,000 rows at claim-only probe 0.5053)
- **Contamination** - CLEAR, rule-generated over already-admitted documents
- **Already covered by** - the misbind family installs the row/column analogue at 0.9603/0.9920; whether that transfers to prose subgroups is unmeasured

## Ranking

Expected arena movement per unit of build cost. The arithmetic throughout: pubmedqa is exactly 10% of the arena mean, so +0.05 on the subset is +0.005 on the headline.

1. **assert_vs_infer** - largest mass (50.3%), a measured deficit against word counting, and the lowest build cost on the board because all three source corpora are already banked, already gated and the operator is deterministic sentence deletion. No acquisition, no generation pass, no GPU before the arm
2. **paraphrase_support** - addresses the other half of the error budget (44 of 88), sources already banked, but the least atomic of the set and its probe reads a profile rather than a skill
3. **aim_vs_finding** - the crispest mechanism and the only one whose probe has a containment baseline at chance by construction, but 15.8% mass and it needs a new corpus acquired, carded and gated first
4. **relation_attested** - the largest per-family deficit but 9.1% mass, a hard structural cap from the read, and a measured history of Wikipedia-register supervision harming this exact subset
5. **scope_bind** - not recommended; n=17

**Build first: assert_vs_infer.** It is the only candidate that is simultaneously the largest failure family, a place where the model measurably underperforms a bag-of-words baseline, and buildable from corpora that are already on disk with passing contamination gates. Its negative-construction operator is also new to the campaign in kind - every banked lane corrupts the CLAIM, this one removes the EVIDENCE - so it tests a direction the mix has never carried.

## Expected arena movement, honestly

- **Point estimate** - pubmedqa +0.03 to +0.07, i.e. arena mean **+0.003 to +0.007**, with wide uncertainty and a real chance of zero
- **Why not larger** - the sentence verifier only has 0.0807 of dilution headroom before the fixed min aggregation, and min is closed (R12: hard_min 0.7355 against best alternative 0.7230; precursor P-B's sentence-exclusion oracle +0.0065). Sentence-level gains reach the item score attenuated
- **Why it might be zero or negative** - `corr(mean, pubmedqa) = -0.859` over 8 recorded reads (R13): pubmedqa has historically moved against the arena mean. The R16-H140 learned readout bought pubmedqa +0.0711 and cost hotpotqa -0.052 and tatqa -0.048 for a net mean of +0.0037. The R11 lane drove pubmedqa below chance outright
- **Replication bar** - same-recipe seed noise on pubmedqa is wide: pooled within-pair seed sd 0.0216 (R16 banked control distribution), and the three-draw spread quoted for this wave is ~0.037. Any claimed move under ~0.05 needs two draws before it is believed
- **Benchmark** - the largest legal pubmedqa move the campaign has banked is the twin protocol's +0.066 (train-serve alignment, 0.6725), and even that sits 0.106 below the 0.7789 ceiling

## Instrument limits

- **77 negatives in 250** - the false-positive rate 0.571 carries binomial SE 0.056; per-family counts below n=20 are coarse and are labelled as such rather than narrated
- **Family counts are small** - relation_not_attested n=15, scope_overextension n=17, contradiction n=10, entity_substitution n=5, false_absence n=2. The two leading families (n=83, n=26) support their readings; the tail does not support build decisions and is not ranked as if it did
- **The rule layer over explanations was not hand-adjudicated end to end** - 24.8% of the 165 explanations fall to `unclassified` under the primary-class assignment. Thirty-five of them were read manually and the recurring shape is relation_not_attested and entity_substitution, so those two families are UNDER-counted in the table above; the regexes were not re-tuned to chase them, and the direction of the bias is recorded rather than corrected
- **Single draw for the sentence-level numbers** - the taxonomy and per-class discrimination are read on h150d1 only, because the h150d2 and h159d1 dumps had not landed when this analysis ran. Item-level control values for h150d2 (0.6298) are taken from the H161 log. The +0.0040 model-over-lexical margin and the +0.0807 dilution are one-draw readings on a subset whose seed sd is 0.0216
- **Alignment is near-total but not total** - 1,301 of 1,314 sentences aligned at similarity >= 0.6 (0.9901); 13 sentences are excluded and their labels are unknown
- **The length baseline is diagnostic, not a proposal** - it is reported to show the verifier adds nothing at item level, and selecting on it would be selection on the benchmark, which the discipline forbids
- **A recorded correction to prior prose** - the R16-H142-T mechanism note describes pubmedqa as "~26 windows/item, the deepest documents in the arena". The measurement here says the opposite: pubmedqa carries exactly 5.00 windows per sentence with zero dispersion and the shortest documents in the arena (mean 372.5 chars, 0.0% over the window). The "~26" is the (sentence, window) pair count, which R16-H141 had already corrected. The twin's gain stands as measured; only that sentence's stated evidence is wrong
