# R14 hypotheses - LENS: READ / SERVING FORMULA

**Scope**: remediations that act through the frozen-weights read - window geometry, evidence-unit construction, per-sentence window-count effects, and read-time tokenization. No training lane is proposed here.

**Lens index**: the orchestrator named this lens "READ / SERVING FORMULA" without a number; ids below use `L4`. Renumber if the campaign assigned a different index - the candidate suffixes `C1..C3` are stable.

**Discipline**: analysis only, Polars throughout, no GPU touched. Every arena figure quoted below was recomputed here from banked artifacts on disk; the four banked per-window score matrices (`R13_dump_h105d1/d2.parquet`, `R13_dump_h108d1/d2.parquet`) reproduce their banked reads exactly before any derived statistic. Arena quantities are ANALYSIS and set no lane sizes or mixes; where an arena measurement was used to shape a proposal, that is stated in the open and the registrable content is moved out of sample.

---

## Verdict, stated first

Three candidates survive, and four read-side ideas that looked plausible from the evidence packs were **killed here for zero cost** before anyone spends GPU time on them.

| id | name | primary target | cost | standing |
|---|---|---|---|---|
| **L4-C1** | EVIDENCE-POOL SIZE DEBIAS | finqa (+0.006 to +0.021), mean HOLD | ~0.8 GPU-h | strongest; sign-consistent on 4/4 banked checkpoints in the free in-sample read |
| **L4-C2** | TOKEN-COMPLETE EVIDENCE UNIT | techqa / finqa / tatqa, mean | ~2 GPU-h | precondition MEASURED and met: 22.9% of scored pairs exceed MAX_LEN today, 0.0% exceed 1024 |
| **L4-C3** | CONTIGUOUS EVIDENCE-UNIT ENLARGEMENT | mean (conjunctive-support loss) | ~2.5 GPU-h | gated behind C2; the one lever aimed at the largest measured structural loss, with an honest counter-indication recorded |

Killed here, free, before registration:

- **claim-numeric-density score debias** (E1's S3 sub-lever) - REFUTED on 4-checkpoint replication: mean negative on all four, finqa sign-inconsistent (h108d1 −0.0057)
- **response sentence-count debias** (the min-side dual of C1) - REFUTED: mean −0.008 to −0.045 on all four checkpoints at every coefficient tried
- **studentized / self-normalised max** (`max − β·median` within a document) - REFUTED: −0.16 mean at β=1, negative at β=0.5 on 3 of 4
- **`truncation="only_second"`** (protect the claim from tail truncation) - NO-OP: the claim is longer than its window on 0.05% of over-length pairs
- **record-boundary snapping** for bracket-JSON tables - NO PRECONDITION: 0.0% of finqa's multi-window windows are tabular, tatqa has 10 multi-window windows in the whole subset
- **numeric-safe sentence splitter** - PRECONDITION TOO WEAK: only 4.0% of finqa's deciding sentences carry a detectable split defect (5.7% of all finqa sentences); the splitter's -0.2500 finqa "cost" in the label-ceiling artifact is an annotation-alignment quantity, not a scoring defect, and ruling 6 keeps it out of bars

**delucionqa is deliberately not a primary target on any candidate.** E2's adjudication stands: with 12 negatives of 184, a 2-draw noise band near ±0.10 and every banked read already above its own faithful-oracle ceiling of 0.6657, a delucionqa bar is unenforceable, and raising delucionqa by firing more readily on partial support is optimizing against that ceiling. delucionqa appears below only as mechanism evidence and as a guardrail.

---

## 1. What the read is today, exactly

From `R8-H101_windowed_read.py`, `R8-H92_decomposed_arena.py` and `R8-H77_unseen_arena.py`:

1. **Split** the response with `re.compile(r"(?<=[.!?])\s+")`; drop parts shorter than `MIN_SENT_CHARS = 25`; cap at `MAX_SENTS = 12`; if fewer than 2 parts survive, score the whole text
2. **Window** every retrieved document into 1,500-char slices at stride 750, final slice flush to the document end (`WIN = 1500`, `STRIDE = 750`, "fixed, not tuned" per the module docstring)
3. **Pool** - every sentence of a response is scored against the *same* pool: all windows of all its documents concatenated. Pool size is therefore constant within a response
4. **Encode** each (sentence, window) pair with `tok(claim, window, truncation=True, max_length=512)` - `truncation=True` is `longest_first`, so the overflow is taken off the window's tail
5. **Aggregate** - sentence score = max over the pool; response score = min over sentences; per-subset metric = response-level AUROC

Four structural facts about this formula that the campaign has not previously separated:

- Step 4 silently shrinks step 2's evidence unit whenever the window tokenizes to more than ~500 subwords. Measured here on the shipped tokenizer: **22.9% of all scored pairs** and **46.4% of techqa's score-deciding pairs** exceed 512 tokens; **0.0% exceed 1024**. Median chars/token by subset ranges from 2.74 (tatqa) and 3.23 (techqa) to 5.17 (emanual) - the geometry is stated in characters, the budget is spent in tokens
- Step 3's max is an order statistic over a pool whose size varies by a factor of 22 across subsets (covidqa 4.00 windows per response, techqa 22.50). Measured here: on sentences with more than one document available, **the winning document carries 4.794 windows against a pool average of 3.876** - the argmax is 24% biased toward the longer document
- Step 5's min is an order statistic over a sentence count that varies 1.17 (hotpotqa) to 6.95 (techqa)
- Step 2's 1,500-char unit is smaller than the support span of one supported sentence in five: the label-ceiling decomposition prices conjunctive support at **−0.1884**, the single largest structural loss in the read, with 20.9% of supported sentences unable to fit all annotated support in any single window

---

## 2. What is already closed in read space

Checked against the canonical log before writing anything below. None of the three candidates re-proposes any of these.

| closed line | verdict | source |
|---|---|---|
| aggregation softening over sentences (softmin τ∈{0.5,1,2,4}, mean, drop-argmin) | P-C NOT FIRED - hard-min 0.7355 beats every alternative; "every aggregation-softening lever - learned or fixed, gated or global" closes | log §P-C |
| sentence exclusion (lexicon, learned abstain head) | P-B KILLED class-level; the drop-argmin oracle reads mean **−0.0359** | log §P-B |
| within-chunk window softening (top-2 consensus mean replaces max) | R13-H124 REFUTED IN SIGN on 4/4 checkpoints; techqa −0.0346 | `R13-H124_result.json` |
| composite/union premise (top-2 windows concatenated) | R13-H125 REFUTED on all three clauses; the pre-registered fire-rate diagnostic showed the composite premise is the argmax *more* often on hallucinated responses on 8/10 subsets | `R13-H125_result_h108d1.json` |
| numeric-surface canonicalization wrapper | R12-H119 REFUTED both directions; "the serving-wrapper canonicalization line closes" | log, R12 ledger |
| token-head-as-primary, head fusion | H102 / H104 / H106 / P-A closed | log |
| window-bag training | KILL binds, ruling 3 | log |
| per-subset serving switches | forbidden by ruling 2's subset-blind condition | log |

**How C1 differs from H124 and P-C.** H124 replaced the within-chunk max with a shrunk statistic (mean of top 2) - the peak is softened. P-C replaced the min over sentences with a softened statistic. C1 changes neither operator: max stays a hard max, min stays a hard min. It applies an additive offset to each *document's* candidate score before the hard max selects among documents. The measured signatures are opposite - H124 lost hagrid on 4/4 and broke the techqa floor; C1 gains finqa on 4/4 and techqa on 4/4.

**How C3 differs from H125.** H125 fabricated adjacency by concatenating two disjoint top-2 windows into a premise that does not exist in the source document, and its fire-rate diagnostic showed that the fabricated premise preferentially certified hallucinated responses. C3 changes only the length of a *contiguous* slice of the real document. No text is assembled that is not adjacent in the source. C3 carries the same fire-rate diagnostic as a pre-registered refutation clause, so if the enlargement buys its gain by leaky certification rather than coverage, it self-refutes.

---

## 3. L4-C1 - EVIDENCE-POOL SIZE DEBIAS

### Mechanism

The sentence score is `max` over a pool of windows. The maximum of K noisy scores drifts upward with K even when the underlying evidence quality is identical - a pure order-statistic effect. In this read K is not a modelling choice, it is `ceil(doc_len / 750)`: it encodes how long the retriever's document happened to be. So document length enters the score through a channel that has nothing to do with support.

Two measurements taken here fix the mechanism as document *selection*, not response calibration:

- **The argmax is biased toward long documents.** On sentences with more than one document in play, the document that wins the max carries 4.794 windows against a pool average of 3.876 (n = 3,567 sentences)
- **Correcting at the response level does nothing; correcting at the document level works.** Because the pool is shared by all sentences of a response, an offset applied to the whole pool is a constant per response and cannot change any argmax. That variant is mean-negative on all four checkpoints (−0.0007 to −0.0058 at α = 0.05). The per-document variant, which *can* re-select which document supplies the max, is mean-positive on three of four and finqa-positive on four of four

The intervention: replace the sentence score

    s_sent = max over all windows w in pool of s(w)

with

    s_sent = max over documents d of [ (max over windows w in d of s(w)) − α · log K_d ]

where `K_d` is the number of windows document `d` was cut into. Subset-blind (a function of document length and the fixed geometry only), idempotent, deterministic, and it ships in the library serving path for every corpus and every future input, satisfying ruling 2's condition. It is zero-parameter at serving time beyond the single scalar α.

At α = 0.05 the correction re-selects the argmax document on **6.13% of sentences**, and exactly where the mechanism predicts: delucionqa 17.6%, techqa 12.2%, finqa 10.3%, expertqa 3.8% - and **exactly 0.00%** on covidqa, pubmedqa and tatqa, whose documents are all single-window. A lever that is an exact no-op on three subsets by construction is not a general score shift.

### Evidence

- **E2 §4.2** - windowing's delucionqa gain is class-asymmetric because 75.6% of grounded responses sit on a multi-window document against 33.3% of ungrounded ones. E2 names this "a property of how RAGBench retrieved context for delucionqa, not of the model". Window count is a label-correlated nuisance covariate. Confirmed here as a standalone read: **window count alone reads delucionqa AUROC 0.6768** and emanual 0.6326, with no model score involved
- **E1** - finqa's long-evidence half reads 0.5929 against the short half's 0.6903. More windows, worse discrimination - the inflation signature
- **E5 / label ceiling** - "the read rewards leaky scoring": a partial-support entailer's ceiling is 0.9444 against the faithful 0.7560. Anything that lets a longer evidence pool certify more easily is the mechanism that gap describes. C1 pushes against it
- **E4** - resp 103's argmax landed on a window that does not contain the support while the supporting window scored lower; document-level selection is where several of the read's errors live

### Measured effect (ANALYSIS, in-sample, disclosed in full)

Recomputed on all four banked per-window dumps. Deltas are against each checkpoint's own banked read.

| α | checkpoint | Δ mean | Δ finqa | Δ techqa | Δ delucionqa | worst subset |
|---|---|---|---|---|---|---|
| 0.02 | h105d1 | +0.00127 | +0.0061 | +0.0075 | −0.0024 | emanual −0.0042 |
| 0.02 | h105d2 | −0.00062 | +0.0065 | +0.0044 | −0.0082 | emanual −0.0109 |
| 0.02 | h108d1 | +0.00108 | +0.0030 | +0.0059 | −0.0029 | emanual −0.0030 |
| 0.02 | h108d2 | +0.00132 | +0.0111 | +0.0011 | +0.0005 | emanual −0.0067 |
| 0.05 | h105d1 | +0.00231 | +0.0165 | +0.0201 | −0.0063 | emanual −0.0121 |
| 0.05 | h105d2 | −0.00090 | +0.0261 | +0.0156 | −0.0082 | emanual −0.0303 |
| 0.05 | h108d1 | +0.00087 | +0.0109 | +0.0146 | −0.0102 | delucionqa −0.0102 |
| 0.05 | h108d2 | +0.00230 | +0.0280 | +0.0013 | +0.0039 | emanual −0.0157 |

finqa is positive on **8 of 8** rows; techqa on **8 of 8**. The arena mean is a HOLD (H105 pair +0.0003 at α=0.02, +0.0007 at α=0.05; H108 pair +0.0012 and +0.0016). delucionqa falls on 6 of 8 - the expected sign, since delucionqa is the subset where window count is most label-correlated and where every banked read already sits above its faithful ceiling.

### Legality and the discipline problem, stated plainly

The table above is an arena measurement. Selecting α from it would be arena-fitting and would void the lever. Two things keep the registration honest:

1. **α is derived, not tuned.** It is the measured order-statistic drift of this model on legal in-domain evidence: take RAGTruth-train and public-mix documents (no RAGBench, no arena), cut each document at the shipped geometry, vary K by varying stride only so that total content is held fixed, and regress the mean sentence-level max score on log K at fixed label. The slope is α. One GPU pass, no arena input, no label from the arena
2. **The registrable adjudication moves out of sample.** The four dumps above are spent. The pre-registered confirmation runs on checkpoints absent from the dump set: **DR-control draw 1, DR-control draw 2 and the H117 margin draw 1** - three checkpoints, ~0.5 GPU-h of fresh per-window dumps, adjudicated at the α that step 1 produced, before any adoption

The sensitivity disclosure is itself part of the argument: the conclusion (finqa up, mean HOLD, emanual the payer) is stable across a 4x range of α, so the derivation only has to land inside 0.02-0.08, not on a knife edge.

### Kill-gate (cheap, before any GPU arm)

Free, on the banked dumps, already run and passed:

- **Selection-bias precondition**: mean window count of the argmax document must exceed the pool mean by ≥ 10% on multi-document sentences. **Measured 4.794 vs 3.876 = +23.7%. PASS.** Had the argmax been unbiased in K, the mechanism would not exist and the lever dies for zero cost
- **Non-degeneracy**: the correction must re-select the argmax document on ≥ 2% of sentences at the derived α, otherwise it is a no-op wearing a mechanism. **Measured 3.06% at α=0.02, 6.13% at α=0.05. PASS**
- **Response-level control** (the falsifier): the response-level variant, which cannot change any argmax, must NOT reproduce the gain - otherwise the effect is a global score shift, not document re-selection. **Measured mean-negative on 4/4. PASS - the mechanism is confirmed as selection**

Then ~0.3 GPU-h for the legal-data α fit. If the fitted slope is ≤ 0 or > 0.15, the order-statistic account is wrong for this model and the lane closes without an arena read.

### Bar (pre-registered, blind, out of sample)

Adjudicated on the three fresh checkpoints (DR-control d1, DR-control d2, H117-margin d1) at the legally derived α, deterministic paired reads - ruling 9 permits tight guards on zero-draw-noise reads.

- **ADMIT** - finqa > 0 on **3 of 3** checkpoints AND finqa mean over the three ≥ **+0.005** AND arena mean ≥ **−0.002** on every checkpoint AND no subset ≤ **−0.035** on any checkpoint AND `gold_full` ≥ its banked value − 0.005 on both H105 draws under the amended read
- **REFUTE** - finqa mean over the three < +0.005 with no sign disagreement, or arena mean between −0.005 and −0.002 on any checkpoint
- **KILL** - finqa negative on any of the three, or arena mean < −0.005 on any checkpoint, or any subset ≤ −0.05

Mean-HOLD rather than mean-gain is the correct shape per ruling 7: this is a subset-targeted lever whose primary is finqa.

### Cost

**~0.8 GPU-h** - ~0.3 for the legal-data α fit, ~0.5 for three fresh per-window dumps. The read itself is CPU arithmetic on the dumps.

### Risks recorded

- The gain is concentrated on finqa and techqa, and E1 documents that a meaningful fraction of finqa AUROC is a verbosity prior (length alone reads 0.6958). C1 does not touch sentence length, so it is not that confound - but the finqa magnitude should not be read as grounding capability
- emanual pays consistently (−0.003 to −0.030). emanual is one of E3's clean instruments (11% noise), so the loss is real, not seed. It is inside the −0.035 guard at α ≤ 0.05 and would breach it above
- The lever's benefit is small in absolute terms (mean HOLD). Its value is that it costs under one GPU-hour and it is the only read change measured sign-consistent on four checkpoints since the windowed read itself

---

## 4. L4-C2 - TOKEN-COMPLETE EVIDENCE UNIT

### Mechanism

The read declares a 1,500-character evidence unit and then encodes it under `max_length=512`. Because `truncation=True` removes overflow from the longer member of the pair, the *tail of the window* is discarded. The discarded amount is a function of the register's token density, not of anything the read chose: 1,500 characters of prose is ~290 subwords, 1,500 characters of serialized tables or logs is 460-550. The scorer's actual evidence unit is therefore smaller than the geometry claims, and it is smallest exactly where the campaign's hardest subsets live.

The fix is one number: `max_length` 512 → 1024. Nothing else moves - same checkpoint, same splitter, same 1,500/750 geometry, same max, same min. The trunk carries `max_position_embeddings` 8192 with RoPE at theta 160,000 and no learned position embeddings (`position_embedding_type: sans_pos`), so 1024 is deep inside the pretrained positional range: **zero new parameters, no extrapolation**.

### Evidence

- **E5's new measurement**, independently reproduced here with the shipped tokenizer on `R12-H121_gateA_scores.parquet`:

| subset | deciding pairs | median tokens | > 512 | > 1024 | median chars/token | median chars lost when truncated |
|---|---|---|---|---|---|---|
| techqa | 250 | 493 | **46.4%** | 0.0% | 3.23 | 237 |
| finqa | 250 | 380 | 9.6% | 0.0% | 4.01 | 193 |
| tatqa | 250 | 223 | 8.0% | 0.0% | 2.74 | 168 |
| expertqa | 203 | 247 | 0.5% | 0.0% | 4.83 | 143 |
| six others | - | 114-313 | 0.0% | 0.0% | 4.5-5.2 | - |

- Pooled over all scored pairs (8,000-row sample): **22.9% exceed 512 tokens, 0.0% exceed 1024**. The fix is complete, not partial - there is no residual truncation left at 1024 anywhere in the arena
- **E5's framing holds**: the mechanism is a property of the window size and the tokenizer, independent of the arena and of any label. The arena figures quantify exposure; they do not motivate the fix
- **The label ceiling** prices window truncation at 0.0000 loss - but that diagnostic measured *geometric* window coverage, not the tokenizer's own truncation of each window. This is a distinct and previously unmeasured loss channel
- **Family precedent**: coverage fixes are the family that produced the campaign's largest deterministic lift (H101 windowing, +0.0142 / +0.0180 on frozen weights)

### Legality

Subset-blind by construction - `max_length` is a single serving constant applied to every input. Idempotent. It changes no text, so it is not a canonicalization wrapper and does not touch the closed H119 line. It ships in the library serving path identically for every corpus.

### Kill-gate (already run, CPU, zero cost)

Tokenize every score-deciding pair in the banked dump with the shipped tokenizer and measure the over-budget share.

- **KILL if** fewer than 5% of deciding pairs exceed 512 tokens on every subset - no truncation exists, no fix needed
- **KILL if** a material share still exceeds 1024 - the move is a partial patch and the geometry needs redesign instead
- **Measured**: 46.4% / 9.6% / 8.0% over 512 on techqa / finqa / tatqa; **0.0% over 1024 everywhere**. **PASS on both clauses.**

A second free check, also run: `truncation="only_second"` was considered as a companion fix to protect the claim from being trimmed. The claim is longer than its window on **0.05%** of over-length pairs - the change is a no-op and is dropped.

### Bar (pre-registered, blind)

Deterministic paired reads on the four banked checkpoints (H105 d1/d2, H108 d1/d2), each amended read compared against that checkpoint's own banked read. Targets are named from the token-density measurement on evidence text only - no arena label enters the target selection.

- **ADMIT** - arena mean ≥ **+0.003** on both pair means with sign agreement on all four checkpoints (the H119 precedent bar), AND the token-dense group (techqa, finqa, tatqa) collectively ≥ **+0.010** with techqa > 0 on at least 3 of 4, AND no subset ≤ **−0.020** on any checkpoint, AND `gold_full` ≥ banked − 0.005 and RAGTruth EN ≥ banked − 0.005 on both H105 draws
- **REFUTE** - mean between −0.002 and +0.003, or sign disagreement within a pair
- **KILL** - arena mean < −0.002 on either pair mean, or techqa negative on 3 or more checkpoints. A coverage fix that loses the register it uncovers has a false mechanism

The prose subsets (covidqa, pubmedqa, hotpotqa, hagrid, emanual, delucionqa) have 0.0% deciding-pair truncation and must move by < 0.002 - a **subset-blind-and-harmless check** in the H119 mould. Movement there would indicate a misconfigured read, not an effect.

### Cost

**~2 GPU-h** - four arena reads at roughly 2x the current per-pair cost on the 23% of pairs that are currently truncated (dynamic padding means short pairs are unaffected). Serving-cost note for adoption: 2/3 of layers use 128-token local attention, so doubling the budget roughly doubles their cost and quadruples attention in the 8 global layers, applied only to the long tail of pairs.

### Risks recorded

- Longer context can hurt: H101's own record shows finqa moved −0.0019 / −0.0267 / −0.0824 across checkpoints when more table text came into view - "more table text in view gives the scorer more numbers to mishandle". The bar's KILL clause is written to catch exactly that
- The model was trained at MAX_LEN 512 (ruling 7's hardware contract). Reading at 1024 is a mild train/serve length mismatch. RoPE at theta 160,000 makes it positionally safe, but the model has never seen a 900-token pair; this is the single largest reason the bar is set at only +0.003
- If C2 admits, C1's α should be re-derived on the amended read before C1 is adjudicated, since the window's effective content changes

---

## 5. L4-C3 - CONTIGUOUS EVIDENCE-UNIT ENLARGEMENT

### Mechanism

The largest measured structural loss in the read is conjunctive support: **−0.1884**, with 20.9% of supported sentences unable to fit all their annotated support inside any single 1,500-char window and 20.0% drawing support from more than one document. A sentence whose support is split across a window boundary can never score high, no matter how good the entailment model is, because no premise the model sees contains the whole support.

H125 attacked this by *assembling* a premise from two disjoint top-2 windows, and was refuted: the fabricated premise fired preferentially on hallucinated responses on 8 of 10 subsets. The failure was adjacency fabrication, not unit size. C3 keeps every premise a genuine contiguous slice of the source document and only makes it longer:

**window 1,500 → 2,250 chars at unchanged stride 750, MAX_LEN 1024.**

Holding the stride constant is deliberate - it isolates the single variable "how much contiguous document does one premise carry" from the confound "how many premises are there". Measured window counts barely move (delucionqa 1.78 → 1.39, techqa 4.50 → 3.67, finqa 1.82 → 1.48; six subsets unchanged at ~1.0), so this is not a repeat of the K-reduction experiment.

### Evidence

- **Label ceiling** - conjunctive support −0.1884, the biggest single structural item, explicitly named in the log as "the single biggest structural lever on the board"
- **E4** - finqa resp 217 (`46.7` present in the response's window pool but not in the window that won the argmax) and resp 191 (three addends present, the sum absent) are cross-window aggregation failures by category; delucionqa's "summary or aggregation sentence spanning multiple windows" is 3 of its 15 worst supported items
- **E4's cross-cutting finding** - "an assertion whose atoms are all present and whose composition is not" is the shared failure of finqa's derived numbers and delucionqa's partial conjuncts. A larger contiguous premise is the read-side half of that; H108's lane is the training-side half
- **Geometry, measured here** - documents longer than 1,500 chars: techqa 83.3%, delucionqa 39.5%, finqa 33.4%, emanual 16.2%, expertqa 15.7%; the other five are ≤ 4%. The lever can only act on five subsets, and is a mathematical no-op on the rest

### Legality

Subset-blind (one geometry constant for every input), deterministic, no text assembled that is not adjacent in the source document. Gated behind C2 because 2,250 chars at the arena's densest observed density (2.74 chars/token, tatqa) is 821 evidence tokens and will not fit under MAX_LEN 512. Per ruling 5's read-amendment precedent, adoption serialises: C3 is read on top of the C2-amended read, never against the unamended one.

### Kill-gate (cheap, before any GPU arm)

Two clauses, both free on the banked dump plus the shipped tokenizer:

- **Geometric non-no-op**: recompute the window grid at 2,250/750 and require that ≥ 20% of score-deciding pairs change their window content. If the enlargement does not reach the pairs that decide scores, it cannot move the metric. (Multi-window documents carry 83.3% of techqa, 39.5% of delucionqa, 33.4% of finqa documents, so this is expected to pass on the geometry alone - it is a guard against an off-by-one in the grid, and against the possibility that deciding pairs concentrate on short documents)
- **Token-fit**: ≥ 99% of pairs at the enlarged window must fit under MAX_LEN 1024. Worst-case arithmetic from the measured densities: 2,250 / 2.74 = 821 evidence tokens, plus a p99 claim of 140 tokens = 961. **Projected PASS**, to be confirmed exactly before the read
- **KILL** if either clause fails - the first means no mechanism, the second means the enlargement needs MAX_LEN 1536 and must be re-priced

### Bar (pre-registered, blind)

Deterministic paired reads on the four banked checkpoints, against each checkpoint's **C2-amended** read.

- **ADMIT** - arena mean ≥ **+0.003** on both pair means with sign agreement on all four, AND no subset ≤ **−0.030** on any checkpoint, AND `gold_full` ≥ banked − 0.005 on both H105 draws
- **Pre-registered independent refutation** (carried over from H125, fires regardless of AUROC): compute the share of deciding pairs whose argmax window is an *enlarged* window, split by response label. If the enlarged window is the argmax at a higher rate on hallucinated responses than on grounded ones on 6 or more subsets, the enlargement is buying its gain by leaky certification and is **REFUTED on mechanism** even if the AUROC bar fires
- **KILL** - arena mean < −0.002 on either pair mean, or delucionqa/emanual ≤ −0.030 (the two subsets whose windowing gain E2 shows is real and would be partly consumed)

### Cost

**~2.5 GPU-h** - four arena reads at ~1.5x tokens per pair on a slightly reduced pair count, plus the fire-rate diagnostic. Runs only if C2 admits.

### Risks recorded, including a counter-indication

- **Counter-indication, measured here.** A free probe on the banked dump - restricting the read to the stride-1500 subgrid, i.e. halving the window count at unchanged window size and unchanged union coverage - costs **−0.0057 on the arena mean** (emanual −0.0400, techqa −0.0239, delucionqa −0.0136, against finqa +0.0207 and tatqa +0.0058). Enlargement is a different move (it grows the unit rather than thinning the grid, and holds the stride) but it does reduce window counts by 15-25%, and that component has a measured negative sign. This is the reason C3 is ranked third and gated
- E2's S1 is the strongest replicated delucionqa mechanism (windowing 10/10 sign-positive, mean +0.0555). Enlargement partially reverses the geometry that produced it. The −0.030 delucionqa/emanual guard is there for that, and delucionqa is explicitly not a target
- The enlargement gives the scorer more text per premise, which is the H101 finqa penalty mechanism ("more table text in view gives the scorer more numbers to mishandle")

---

## 6. Sequencing, composition, and what to run first

1. **C1's kill-gates are already passed and its arena analysis is already spent** - the only outstanding work is ~0.3 GPU-h for the legal-data α derivation and ~0.5 GPU-h for three out-of-sample dumps. Run it first: it is the cheapest item on the board and the only read change with 4/4 checkpoint sign agreement
2. **C2 next** at ~2 GPU-h, kill-gate already passed on measurement
3. **C3 only if C2 admits**, read on top of the C2-amended read per ruling 5

**Composition.** C1 and C2 are near-orthogonal: C1 changes which document supplies the max, C2 changes what each window contains. C1 and C3 overlap - the enlargement reduces window counts, which shrinks the pool-size bias C1 corrects, so C1's α must be re-derived after any C3 adoption and its measured effect will attenuate. Do not sum their deltas.

**Combined optimistic arithmetic, stated honestly**: C1 mean HOLD (+0.000 to +0.002), C2 unknown with a +0.003 bar, C3 unknown with a +0.003 bar. Even on the optimistic branch the read lens contributes under +0.010 of arena mean against the 0.0369 gap to the 0.74 goal. The read lens's value in R14 is that it is cheap, deterministic and finqa-positive - not that it closes the gap.

---

## 7. What this lens declines to propose, and why

- **Any delucionqa-primary bar.** E2's DECLINE ON MEASURABILITY verdict is accepted in full: 12 negatives, bootstrap CI width 0.2749, seed sigma 0.0432 against an analytic AUROC SE of 0.0485, zero of 14 trained configurations moving it past 2 sigma, and every banked read already above its faithful-oracle ceiling of 0.6657. delucionqa is used here only as mechanism evidence (E2 §4.2's class-asymmetric window exposure is C1's founding measurement) and as a guardrail
- **Any read change that raises a subset by firing more readily on partial support.** The label ceiling prices the leaky entailer at 0.9444 against the faithful 0.7560. C1 pushes against that gradient; C3 carries H125's fire-rate diagnostic as an independent refutation clause precisely so it cannot pass by exploiting it
- **Any per-subset switch, threshold or geometry.** Ruling 2's subset-blind condition binds all three candidates; the only subset-conditional behaviour anywhere above is arithmetic no-ops on subsets whose documents are single-window
- **Aggregation softening in any form.** P-C closed the class and C1 is not a member: both hard operators survive unchanged

---

## Reproduction

All figures recomputed with:

```
cd /home/lab/workspace/private/ai-assistants/groundrails
uv run python   # Polars, CPU only; tokenizer from models/R9-H105-mmbert-dann-clean
```

Artifacts read: `R12-H121_gateA_scores.parquet`, `R13_dump_h105d1.parquet`, `R13_dump_h105d2.parquet`, `R13_dump_h108d1.parquet`, `R13_dump_h108d2.parquet`, `R9-H105_windowed_result.json`, `R9-H105_draw2_windowed_result.json`, `R10-H108_lane_draw{1,2}_windowed_result.json`, `DR_lane_draw{1,2}_control_windowed_result.json`, `R8-H101_windowed_read.py`, `R8-H92_decomposed_arena.py`, `R8-H77_unseen_arena.py`, `R12-H119_windowed_read.py`, `models/R9-H105-mmbert-dann-clean/`, and `docs/experiments/semantic-grounding-experiments.md` (rulings at lines 2477-2532, P-B/P-C at 2140-2160, H119/H124/H125 verdicts at 2585-2605).

Baseline reproduction confirmed before any derived statistic: H105 draw 1 windowed mean 0.70471, draw 2 0.70151, pair mean **0.70311**; per-subset pair means covidqa 0.7878, delucionqa 0.8166, emanual 0.6976, expertqa 0.7728, finqa 0.6333, hagrid 0.6340, hotpotqa 0.6667, pubmedqa 0.6063, tatqa 0.7320, techqa 0.6840 - matching E3's baseline row exactly.
