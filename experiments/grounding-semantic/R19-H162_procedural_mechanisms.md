# R19-H162 Procedural-Register Mechanism Dissection (emanual + techqa)

ANALYSIS ONLY. Executor M4 of the R19-H162 mechanism-dissection wave. Targets the two
technical-documentation subsets of the blind arena - `emanual` (consumer-electronics manual QA,
flagship 2-draw 0.6780) and `techqa` (enterprise technical-support QA, flagship 2-draw 0.7335).
Nothing here trains, tunes, or selects on arena statistics; no GPU was used. Every model score is
read from the banked R19-H161 per-pair logit dump, whose own positive control reproduced the banked
windowed AUROCs to <= 1e-3.

## Headline

Two findings dominate, and the first changes the bar for every lane proposed below.

- **The trained cross-encoder has no measurable advantage over token containment on either subset** -
  a deterministic surface scorer (fraction of the sentence's content tokens present in the window),
  aggregated MAX-over-windows then MIN-over-sentences exactly as the model is, reads emanual
  **0.7763** against the model's 0.6973 and techqa **0.7462** against the model's 0.7361. Arena-wide
  the model still leads a mean 0.7144 to 0.6582, and on the third procedural subset `delucionqa` it
  leads 0.8009 to 0.5889 (+0.2120, about 4 instrument SE). The procedural register is where the
  semantic tier stops paying
- **The max-over-many-windows inflation hypothesis is REFUTED on techqa, decisively** - the sentence
  max does not rise with window count (Spearman -0.045), window count alone reads AUROC 0.4391
  (anti-predictive), and randomly capping each sentence's window set costs AUROC monotonically:
  K=40 -0.0007, K=20 -0.0126, K=10 -0.0706, K=5 -0.1295, K=3 -0.1562. Every extra window is net
  evidence, not net noise. The same capping test costs emanual -0.0544 at K=3 and nothing at K>=10

## Instruments

- **techqa is the campaign's best-powered blind subset** - 250 items, 141 supported / **109
  unsupported**, AUROC SE 0.0310, seed SD ~0.0136; 1,737 response sentences, 5 documents per item at
  a mean 3,730 chars, 39,058 sentence-window pairs
- **emanual is a weak instrument and must be read as one** - 132 items, 118 supported / **14
  unsupported**, AUROC SE **0.0654**, seed SD ~0.0285; any claimed movement below ~0.06 on emanual
  is unresolvable, and the H147 re-pricing of its hold to control - 0.12 already records this
- **Window geometry, corrected** - the "156 windows per item" figure counts (sentence, window) pairs.
  The number the MAX actually runs over is **26.8 windows per sentence on techqa** (median 26, max
  82) against 4.7 on emanual and 6.4 on delucionqa. The same conflation was recorded as a process
  failure in the H141 autopsy
- **Error split at the in-sample macro-F1-optimal threshold** (the H147/H157 stated choice; the
  threshold-free AUROC is reported alongside and nothing is tuned on it) - techqa 72 errors of 250:
  **48 false positives of 109 negatives** (rate 0.440, binomial SE 0.048) and 24 false negatives of
  141 (0.170, SE 0.032); emanual 18 errors of 132: **10 false positives of 14 negatives** (0.714,
  SE 0.121) and 8 false negatives of 118 (0.068, SE 0.023)

## What the register actually asks

Read from the frozen gate samples, both subsets ask the same four questions of a claim, and none of
them is a paraphrase judgement.

- **Which product and which release** - techqa evidence is IBM technotes, APAR records and security
  bulletins, each scoped to a named product at named version ranges; a claim is supported only if the
  fact it states is bound to the release the evidence states it for
- **Under which heading** - emanual evidence is a manual sectioned by task headings ("Creating a new
  account", "Updating through a USB device", "Using Timeshift"); an individual step can be verbatim
  present and still belong to a different procedure
- **Under which precondition** - both registers gate instructions on device state, model, region or
  version ("when the TV is turned off", "may not be supported depending on the model or geographical
  area", "supported starting with Cumulative Fix #8")
- **Along which path** - emanual serialises UI navigation as a bare token run with the separators and
  glyphs stripped by extraction ("Settings General System Manager Samsung Account Add Account Create
  Account"); responses render the same path arrowed

## Measured mechanisms

### bind_product_version

Bind a stated capability, fix, requirement or vulnerability to the exact product-and-release
identifier the evidence states it for.

- **Applies to** - techqa (primary, measured), emanual (secondary, model/region applicability),
  delucionqa (candidate: model-year and trim applicability, not measured here)
- **Bottleneck evidence** - techqa items split on whether the response carries a version, CVE or APAR
  identifier: **no identifier (n=124, 49.6%) reads 0.7891; all identifiers present in the evidence
  (n=87, 34.8%) reads 0.6824; some identifier absent from the evidence (n=39, 15.6%) reads 0.6316**.
  The model loses -0.107 to -0.158 AUROC the moment a release identifier is in play, and **126 of 250
  items (50.4%) are in that condition**. In the hardest stratum token containment beats the model
  0.7000 to 0.6316
- **The model tracks identifier presence, not identifier correctness** - sentences whose identifiers
  are all present in their own argmax window score a mean logit **+1.752** (n=180) against **-0.124**
  for sentences whose identifiers are absent (n=116); Spearman of identifier containment against the
  model score is 0.512, statistically indistinguishable from plain token containment at 0.577
- **Reading** - the false negatives are almost all applicability statements at high token containment:
  "The WebSphere Application Server Version 7 ships with the WebSphere MQ 7.0 Resource Adapter"
  (items 83, 215, containment 0.90, scores -1.50 and -1.65), "For Netcool/OMNIbus V7.4.0, the legacy
  Socket Gateway to be used is nco-g-socket-10_0" (item 82, 0.75, -1.52), "ITCAM for J2EE v7.1.1.0 did
  not officially support RHEL7" (item 228, 0.75, -1.18). All tokens are present; the binding is not
  confirmed
- **Probe design** - held-out synthetic minimal pairs over compatibility and advisory blocks listing
  three or more products at three or more release ranges. Positive restates the true (product,
  release, property) triple; negative swaps the product or shifts the release range to a sibling
  present in the same block, leaving the token multiset otherwise identical. Target >= 1,200 pairs,
  >= 600 per family (product swap, release shift) per the H149 power ruling. **Chance level 0.50**
- **Lane candidate** - **NVD / CVE JSON feeds** (NIST, US Government work, public domain under
  17 U.S.C. 105 and explicitly free to use): every record pairs a free-text description with CPE
  applicability statements carrying vendor, product and machine-readable version ranges. Negatives
  are generated by field swap, exactly the graduated `quant_misbind` construction on a new axis.
  Secondary supply: Debian Security Advisories and Ubuntu USN (public), OSV.dev (CC BY 4.0 aggregate,
  per-source terms to be checked at build)
- **Contamination** - **CLEAR**. NVD is not a RAGBench source corpus nor a derivative of one; TechQA
  is IBM technotes, a separate document population. CVE identifiers will co-occur across both, which
  is entity overlap, not document overlap - a CVE-id overlap census and a document-disjointness check
  are required at lane build, on the R10-H107 precedent that permits register-only overlap
- **Already covered by** - nothing banked. The graduated `quant_misbind` family (R17-H146) installs
  value-to-column and value-to-row binding over tables at 0.9555 / 0.9908; this is the same skill with
  a release identifier as the binding axis and prose rather than a table as the carrier

### bind_path_segment

Bind each segment of a UI navigation path to its level in the menu hierarchy, across a rendering
change - arrowed path in the claim against a bare token run in the evidence.

- **Applies to** - emanual (primary, measured), delucionqa (candidate: infotainment menus)
- **Bottleneck evidence** - 31 of 748 emanual sentences (4.1%) render an arrowed path; their mean
  model logit is **+0.170** against **+1.519** for every other sentence. The 19 items that contain one
  (14.4% of emanual) read a within-stratum AUROC of **0.583** against 0.711 for the 113 that do not
- **Reading, both directions** - item 131 (supported, scored -3.095 at token containment 0.909) states
  "Settings > General > System Manager > Samsung Account > Add Account > Create Account" against
  evidence carrying that exact bare run; item 15 (unsupported, ranked at the 73rd model percentile
  against the 28th lexical percentile) decomposes the manual's "Network Reset" path into "Select
  Network / Choose Network Reset", transposing two levels while preserving the token multiset. The
  model misses the correct binding and credits the transposed one
- **Probe design** - rule-generated. A menu tree of depth 3-5 is serialised bare into the evidence and
  arrowed into the claim; positives restate the true path, negatives transpose two adjacent segments,
  drop a level, or substitute a sibling segment from the same tree. The transposition family holds the
  token multiset exactly constant, so no surface feature can separate it. **Chance level 0.50**
- **Lane candidate** - **rule-based generator, no corpus required** - the strongest structural fit to
  the `quant_misbind` pattern in this memo. For lexical realism, hierarchical settings vocabularies can
  be drawn from GNOME user documentation (CC BY-SA 3.0), LibreOffice help (MPL 2.0) or the Debian
  Administrator's Handbook (GPL-2 / CC BY-SA 3.0)
- **Contamination** - **CLEAR** by construction; a generator has no source population, and none of the
  named realism corpora is a RAGBench source or derivative
- **Already covered by** - nothing banked

### bind_step_to_procedure

Bind an instruction step to the procedure heading it belongs under, when the same step text appears
verbatim under a different heading.

- **Applies to** - emanual (primary), delucionqa (candidate - the same manual structure), techqa (weak)
- **Bottleneck evidence** - **underpowered, labelled as such**. 4 of emanual's 10 false positives read
  as this shape: item 65 answers "how do I create a SmartThings account" with the device-registration
  procedure, every step of which is verbatim in the manual (sentence scores +2.76 to +3.79) under the
  wrong heading, and the item's min lands at -0.462 against a -2.963 operating point; item 114 answers
  "how do I check scheduled viewings" with the add-a-viewing path; item 15 is the transposed network
  reset. At 14 negatives the count carries a binomial SE of 0.121 and cannot be resolved further on
  this instrument
- **Prior read that bears on it** - R17-H148 measured `misbound_step` at **0.8697** on the banked clean
  checkpoint, that is step-ORDER binding is already installed, while `misbound_value` read **0.6243**
  (SE 0.0199, 390 pairs). Neither family probed goal binding, which is the shape read here
- **Probe design** - procedure blocks carrying three or more headings with three to five steps each.
  Positive restates a step together with its own heading's goal; negative restates the identical step
  under a sibling heading's goal. Step-number families excluded per the H148 reopening condition.
  **Chance level 0.50**
- **Lane candidate** - `army-tm` (public domain, 17 U.S.C. 105) plus `faa-amt` (public domain) are the
  registered supply and **remain blocked**: the crawl still holds 135 of 1,766 PDFs, all lubrication
  orders, zero numbered-step operator manuals, against H148's measured 429 procedural blocks / 102
  documents. `multidoc2dial` (488 US government-service documents, grounded spans) is now on disk at
  `data/external/datasets/dataset-multidoc2dial.zip`, closing the "no offline shards" half of H148's
  supply block - but it is the same corpus whose broad-register import as R10-H107's `proc_gov` leg was
  refuted at -0.0384 arena mean
- **Contamination** - **CLEAR** by construction; army-tm, FAA and multidoc2dial share no documents with
  the Samsung TV manual or IBM technotes
- **Already covered by** - partially. R10-H107 imported this register broadly and was refuted; R17-H148
  probed the step-number family and killed at gate. Neither addressed goal binding

### condition_applicability

Bind an instruction to the device state, model, region or release precondition under which the
evidence states it holds.

- **Applies to** - emanual, delucionqa (candidate), techqa (as the applicability half of
  bind_product_version)
- **Bottleneck evidence** - **a lead, not a resolved measurement**. 2 of emanual's 10 false positives:
  item 108 asserts "To enter Ambient Mode when the TV is already turned on, press the button" where the
  manual documents only the TV-off case (score -2.001, above the -2.963 operating point); item 43
  inverts a conditional - the manual states that selecting Dolby Digital+ on a receiver that does not
  support it CAUSES no sound, the response prescribes it as the fix for no sound (score -1.209 at token
  containment **1.000**). At 14 negatives this is 2 items and cannot carry a lane on its own
- **Probe design** - instruction plus precondition pairs; negatives attach the instruction to a sibling
  precondition from the same block or invert the conditional's direction. **Chance level 0.50**
- **Lane candidate** - the same army-tm / FAA warning-and-caution supply, therefore the same supply
  block; or rule-generated preconditions over synthetic device-state taxonomies
- **Already covered by** - nothing. H148's `misbound_value` read of 0.6243 is the closest prior signal
  and points the same way, 1.3 SE below its own bar

### pointer_answer_credit

Credit a response that names where the answer lives ("this is described in security bulletin X") as
though it stated the fact.

- **Applies to** - techqa (primary, measured); absent from emanual at usable mass
- **Bottleneck evidence** - 51 of 250 techqa items (20.4%) contain a pointer sentence. Their mean item
  logit is **+0.020** against **-0.635** for the rest, a +0.655 lift, and within-stratum AUROC is 0.686
  against 0.743. **10 of the 48 techqa false positives have a pointer sentence as their SINKING
  sentence** (rule-matched: items 13, 25, 37, 113, 115, 181, 196, 198, 199, 222), and the subset's
  single largest false positive (item 181, sinking sentence +3.84 at containment 0.79) is one
- **Honest caveat** - part of this is techqa's label regime rather than a verification skill: a response
  that points instead of answering is unhelpful, and the annotators marked it unsupported. A lane that
  teaches "a pointer is not support" is teaching a labelling convention as much as a skill
- **Probe design** - positives state the fact; negatives assert that the fact is documented in a named
  sibling source that does not contain it. **Chance level 0.50**
- **Lane candidate** - rule-generated over any titled document collection; NVD advisories supply both
  titles and contents, so this rides the bind_product_version generator at near-zero marginal cost
- **Contamination** - **CLEAR**
- **Already covered by** - nothing banked

### discourse_frame_sink

A response's contentless preamble or closing recap decides the item's MIN, so the item score reports a
sentence that carries no proposition to verify.

- **Applies to** - emanual (measured, strong), techqa (measured, weak at item level), delucionqa
  (candidate - every subset's responses are LLM-authored)
- **Bottleneck evidence, emanual** - recap sentences are 24 of 748 sentences (3.2%) but are the item's
  sinking sentence **58.3%** of the time against 16.3% for every other sentence, a **3.6x lift**; their
  mean logit is -1.132 against +1.549. The 24 items that end in a recap read a within-stratum AUROC of
  **0.550 - chance** - against 0.715 for the 108 that do not. Preambles are 10.6% of sentences and sink
  27.9% of the time against 16.4%
- **Bottleneck evidence, techqa** - the sentence-level lift replicates (recaps sink 23.9% against 14.1%,
  preambles 20.8% against 13.6%) but the item-level stratum effect vanishes (0.734 against 0.737), so
  this is an emanual mechanism, not a techqa one
- **Disposition - NOT A DATA LANE** - this is a read-protocol property, and the H151 wave closed
  serving-read changes with MAX standing as PRIMARY. It joins emanual's list-half (H147) and hagrid's
  bare-assertion half (H149) as a diagnosed, unfunded deficit. Recorded because it bounds what any
  emanual lane can achieve: 18% of emanual items are decided by a sentence with nothing to verify

## Provenance concentration - diagnostic, not a lever

A label-free geometric feature reaches or beats the model on both subsets, which is the same finding as
the lexical ceiling arriving by a different route.

- **The feature** - the share of an item's sentences whose argmax window falls in the item's modal
  document. Supported emanual answers draw on 1.653 documents and concentrate 0.836 of their sentences
  on one; unsupported answers draw on 2.286 and concentrate 0.690. On techqa the split is 2.149 against
  3.009 documents and 0.752 against 0.595 concentration
- **Alone it reads AUROC 0.7228 on emanual** (model 0.6973) and 0.6993 on techqa (model 0.7361); on
  delucionqa it reads 0.5962 against the model's 0.8009
- **Not proposed as a lever** - H147 already killed retrieval geometry as an emanual direction, and
  selecting a serving feature because it moves an arena number is exactly what the H141 discipline
  forbids. It is recorded as corroboration that emanual's and techqa's model scores are carrying less
  verification signal than their AUROCs suggest
- **Supporting geometry** - **47.4% of techqa sentences** have their model-argmax window in a different
  document from their lexically-best window (emanual 22.6%, delucionqa 39.0%), so on techqa the model's
  "best shot at support" is off the most-overlapping source about half the time

## Ranking

Ranked by expected arena movement per unit of build cost, with the instrument's ability to verify the
result treated as part of the cost.

| Rank | Mechanism | Mass | Measured deficit | Supply | Verifiable on arena |
|---|---|---|---|---|---|
| 1 | `bind_product_version` | 126 of 250 techqa items (50.4%) | -0.107 to -0.158 AUROC between strata | NVD, public domain, machine-readable | Yes - techqa SE 0.031, seed SD 0.0136 |
| 2 | `bind_path_segment` | 19 of 132 emanual items (14.4%) | 0.583 within-stratum against 0.711 | Generator, no corpus needed | No - emanual SE 0.065 |
| 3 | `pointer_answer_credit` | 51 of 250 techqa items (20.4%) | +0.655 logit lift, 0.686 against 0.743 | Rides the rank-1 generator | Marginal |
| 4 | `bind_step_to_procedure` | 4 of 10 emanual false positives | Unresolvable at 14 negatives | Blocked - crawl at 135 of 1,766 | No |
| 5 | `condition_applicability` | 2 of 10 emanual false positives | Lead only | Same block as rank 4 | No |
| - | `discourse_frame_sink` | 24 of 132 emanual items (18.2%) | 0.550 within-stratum, chance | Not a data question | n/a |

**Build first: `bind_product_version`.** It is the only mechanism on this lane whose target subset is a
powered instrument, it covers half that subset's items, its deficit is measured at -0.107 to -0.158
AUROC between strata rather than inferred from a handful of read items, its supply is public-domain and
machine-readable at a scale no crawl gates, and its generator is the already-graduated misbind
construction applied to a release identifier instead of a table axis. Ranks 2 and 3 are cheap enough to
ride behind it; ranks 4 and 5 remain supply-blocked exactly where H148 left them.

## Expected arena movement

Stated with its uncertainty, and bounded by the lexical ceiling above.

- **techqa, if bind_product_version installs** - lifting the two identifier strata (n=126, currently
  0.6824 and 0.6316) toward the identifier-free stratum's 0.7891 puts techqa in the region of 0.78-0.79.
  That is **+0.04 to +0.06 on techqa, or +0.004 to +0.006 on the arena mean**, against a seed SD of
  0.0136 - measurable at about 3 SD if it lands
- **The honest discount** - the same lift would take techqa from 0.7335 past a token-containment
  baseline that already reads 0.7462. Any lane that ends below ~0.75 on techqa has not demonstrated a
  semantic contribution, only recovered one. That is the bar this memo recommends the arm be registered
  against, alongside the arena number
- **emanual, on any lane here** - **not verifiable**. A +0.02 to +0.03 movement is the realistic size of
  the path-binding and goal-binding effects, and emanual's instrument SE is 0.0654 with a same-recipe
  seed spread of 0.0285. The probe is the only honest primary; the emanual arena read stays REPORTED, as
  R17-H148 had already registered it
- **Ceiling context** - the faithful-oracle ceiling under the shipped read is techqa 0.8682 and emanual
  0.8160, so techqa carries +0.133 of headroom above the flagship and emanual +0.138

## Shared with delucionqa

- **The register is shared; the failure is not** - delucionqa is the same procedural family and the same
  manual structure, and `bind_step_to_procedure`, `bind_path_segment` and `condition_applicability` all
  plausibly apply to it. But the diagnostic that defines emanual and techqa does **not**: on delucionqa
  the model beats token containment 0.8009 to 0.5889 (+0.2120, about 4 SE), and provenance concentration
  alone reads 0.5962 against the model's 0.8009. Whatever the model is doing on delucionqa, it is not
  what it is doing on the other two
- **On the -0.1025 collapse** - the mechanisms named here do not explain it. That collapse was measured
  on the enriched-mix arm against the flagship 2-draw mean, and nothing in this lane's evidence
  distinguishes an enriched-mix displacement from the standing deficits. Naming a shared mechanism for
  it would be a guess, and it is dropped rather than labelled
- **What would test it** - the one shared candidate worth a measurement is `condition_applicability`,
  because a car manual gates almost every instruction on trim, model year and equipment package. It is
  not measured here and is a lead only

## Caveats

- **emanual is not a reliable instrument** - 14 negatives, AUROC SE 0.0654, same-recipe seed spread
  0.0285. Every emanual count in this memo (10 false positives, the 4/2 mechanism split) carries a
  binomial SE of about 0.12. The mechanism readings are directional; the counts are not resolvable
- **The lexical-ceiling gaps are not individually significant** - emanual +0.0790 is 1.2 instrument SE
  and techqa +0.0101 is 0.3 SE. The defensible claim is "no measurable advantage over token
  containment on these two subsets", not "worse than". The contrast that IS significant is delucionqa's
  +0.2120 in the other direction
- **One checkpoint, one draw** - all measurements read `models/R18-H150-arm-draw1` through the h150d1
  dump. The h150d2 and h159d1 dumps were still writing when this lane ran, so nothing here is
  cross-draw confirmed. The R18-H157 precedent found the finqa taxonomy stable across draws while the
  error SETS were not; the same should be assumed here until the second dump is joined
- **The mechanism taxonomy is a reading, with a rule layer only where a rule exists** - identifier
  binding, path rendering, pointer phrasing and discourse framing are matched by regular expressions
  and their incidence is exact; goal misbinding and conditional applicability were classified by
  reading all 90 error records and carry no rule layer
- **Sentence-class regexes under-count** - the preamble and recap patterns are conservative and will
  miss unmarked framing sentences, so the 3.2% recap share on emanual is a floor, not an estimate
- **The identifier regex is coarse** - it matches dotted version strings, CVE ids and two-letter APAR
  ids, and finds zero identifiers in emanual responses. Emanual's own applicability axis (model names,
  QLED-specific functions, geographical area) is therefore not covered by the stratified measurement
  and is read-only evidence
- **Contamination discipline** - RAGBench source corpora and derivatives are never proposed as training
  data. EManual and TechQA items were read to characterise the task, which the wall permits; the lane
  candidates are NVD, public-domain government manuals, permissively-licensed software documentation,
  and rule-based generators

## Artifacts

- `experiments/grounding-semantic/R19-H162_procedural_autopsy.py` - window-inflation test and geometry
- `experiments/grounding-semantic/R19-H162_procedural_export.py` - error dossier builder
- `experiments/grounding-semantic/R19-H162_procedural_mech.py` - lexical ceiling, identifier binding,
  sentence and item classes
- `experiments/grounding-semantic/R19-H162_procedural_mech2.py` - provenance concentration, identifier
  strata, lexical/model disagreement
- `experiments/grounding-semantic/R19-H162_procedural_mechanisms_summary.py` - assembles the JSON
  deliverable from the measurement files, so no number in it is hand-copied
- `experiments/grounding-semantic/R19-H162_procedural_mechanisms.json` - the lane's JSON deliverable
- `experiments/grounding-semantic/R19-H162_procedural_inflation.json` - window-inflation and geometry
- `experiments/grounding-semantic/R19-H162_procedural_mech.json` - mechanism measurements
- `experiments/grounding-semantic/R19-H162_procedural_mech2.json` - stratified measurements
- `experiments/grounding-semantic/R19-H162_procedural_errors.parquet` / `.txt` - 90 error records with
  sinking sentence, argmax window and surface features
- `experiments/grounding-semantic/R19-H162_procedural_geometry.parquet` - rebuilt sentence text
- `experiments/grounding-semantic/R19-H162_procedural_disagree.txt` - items surface overlap ranks better
- `logs/R19-H162_procedural.log`
