# R14-H130 (R14-A1) - frozen functional form and alpha-derivation procedure

**Status**: written and closed BEFORE any H130 measurement was taken, per binding amendment (i) of R14-A1 ("alpha and the form must be frozen in writing before the three fresh dumps are read, and no re-derivation is permitted after seeing them").

Nothing in this document may be revised once the alpha fit or the three fresh per-window dumps have been produced. If the form below turns out to be a poor choice, the correct outcome is a recorded failure, not an amended form.

---

## 1. The functional form (frozen)

The serving read is unchanged in every respect except one arithmetic step inserted between the per-document max and the cross-document max.

Let a response be split into sentences `s` by the shipped splitter, and let the retrieved evidence be documents `d = 1..D`, each cut into `K_d` windows by the shipped geometry (window 1,500 characters, stride 750, final window flush to the end). Let `score(s, w)` be the frozen model's sigmoid score for sentence `s` against window `w`.

**Incumbent (uncorrected)**

    sent(s)  = max over all windows w of score(s, w)
    resp     = min over sentences s of sent(s)

**Corrected (H130)**

    sent_alpha(s) = max over documents d of [ ( max over w in windows(d) of score(s, w) ) - alpha * ln(K_d) ]
    resp_alpha    = min over sentences s of sent_alpha(s)

Frozen details, all binding:

- `ln` is the natural logarithm. `K_d = 1` therefore contributes exactly zero offset, so single-window documents are untouched by construction
- `K_d` is the window count of document `d` under the **serving** geometry (1,500 / 750), i.e. the `n_win_in_doc` column of the per-window dump. It is not re-derived from any other stride
- The offset is applied **per document, before the cross-document max**. It is not applied per window, not applied after the max over documents, and not applied at response level. (The response-level variant is the pre-registered falsifier and has already been measured mean-negative on 4/4.)
- One global scalar `alpha`. No per-subset alpha, no per-checkpoint alpha, no per-domain alpha, no clipping, no flooring, no interaction with document length in characters
- The sentence set, the sentence labels, the document truncation (`MAX_CHUNKS = 8`), the response-level `min`, the AUROC estimator and the response adherence labels are all byte-identical to the incumbent read. The only change in the whole pipeline is the subtraction above

## 2. The alpha-derivation procedure (frozen)

**Premise being fitted.** `max over K windows` is an order statistic. Holding the *content* of a document fixed and increasing only the number of windows the max is taken over inflates the max. `alpha` is the measured inflation per unit `ln K`, and the correction removes exactly that much.

**Model used for the fit.** The frozen H105 draw-1 checkpoint (`models/R9-H105-mmbert-dann-clean`), the campaign's reference recipe baseline. A single alpha is fitted once on this checkpoint and then applied unchanged to every checkpoint read in section 3. Alpha is a property of the read's order statistic, not of a checkpoint, and re-fitting per checkpoint would be a second free parameter.

**Data - legal, in-domain, arena-free.** Rows drawn from `R9-H105_clean_mix.public_train()` (RAGTruth EN train + its 7 translations, HaluEval, PsiloQA, VitaminC, TabFact). RAGBench is not touched at any point in the fit; no arena document, response, sentence, label or score enters it. Only rows whose evidence chunk exceeds 1,500 characters are eligible, since a document at or below the window size admits no `K > 1`.

> **Amendment A1-a, recorded 2026-08-09, BEFORE any alpha value existed and BEFORE any fresh dump was produced.** The first execution of the fit returned **zero** eligible rows and crashed before computing any slope (`logs/R14_H130_alpha.log`, first attempt). Cause: `public_train()` truncates every evidence chunk to `M59.CFG.chunk_max_chars = 1500` on its way into the trainer, so in the mix as the trainer sees it `K` is identically 1 and the order statistic being fitted does not exist. The fix is to read the same rows with the same filters from the same corpora and **not apply that truncation** - `M59.CFG.chunk_max_chars` is raised before `public_train()` is called, so the row set, the label logic, the filters and the group tags are byte-identical and only the evidence text is full length. This changes the data loader, not the functional form, the estimator, the stride grid, the sample size, the seed, the model or any clause. It is recorded here because the frozen document must reflect what was run.

**Sampling, fixed in advance.** `numpy.random.default_rng(seed = 20260809)`, stratified over the DANN group tags, `N_ITEMS = 1000` eligible rows. An "item" is one (claim, document) pair. The claim text is the mix row's claim, unmodified.

**Stride grid, fixed in advance.** Window width is held at 1,500 characters throughout - only the stride varies, so every grid point sees the same document content and differs only in how many windows the max ranges over:

    STRIDES = (1500, 1000, 750, 500, 375, 250)

The windowing function is the shipped one (`R12-H121_gateA_dump.windows` semantics: starts at `range(0, n - 1500 + 1, stride)`, plus a final flush window when the last start leaves a tail). Grid points that yield a duplicate `K` for an item are retained; the fit is over `(ln K, M)` pairs, and duplicates carry no extra weight beyond their count.

**Response measured.** For item `i` at stride `t`: `M(i, t) = max over the K(i, t) windows of score(claim_i, window)`.

**Estimator.** Within-item ordinary least squares - each item is its own fixed effect, so the fit is on deviations from the item mean and no cross-item difference in document difficulty can enter:

    alpha_hat = sum over (i, t) of  dM(i,t) * dL(i,t)  /  sum over (i, t) of  dL(i,t)^2

where `L(i,t) = ln K(i,t)`, `dL = L - mean_t L(i, .)`, `dM = M - mean_t M(i, .)`. Items whose `K` is constant across the whole grid contribute nothing (zero variance in `L`) and are dropped. The standard error is the clustered-by-item OLS standard error and is reported but is **not** part of any clause.

**Rounding.** `alpha = round(alpha_hat, 3)`. This rounded scalar is the frozen constant used in section 3.

**KILL clause (pre-registered, from the spec).** `alpha_hat <= 0` or `alpha_hat > 0.15` closes the lane with no arena read. The rounding above is applied after the clause is evaluated on `alpha_hat`.

## 3. What is read afterwards, and the bar (restated, unchanged)

Three checkpoints absent from the spent dump set, each dumped fresh at the per-(sentence, window) level with the shipped geometry:

| tag | checkpoint | banked uncorrected read |
|---|---|---|
| drc1 | `models/DR-lane-draw1-control` | `DR_lane_draw1_control_windowed_result.json` |
| drc2 | `models/DR-lane-draw2-control` | `DR_lane_draw2_control_windowed_result.json` |
| mgn1 | `models/DR-lane-draw1-margin` | `DR_lane_draw1_margin_windowed_result.json` |

Each dump is reduced twice on CPU: once with `alpha = 0` (which must reproduce the banked per-subset AUROCs, and is the registered reproduction check) and once with the frozen `alpha`. The reported delta per subset is `AUROC(alpha) - AUROC(0)` **computed on the same dump**, so the comparison is paired at the row level and carries no re-run noise.

**ADMIT** - finqa > 0 on 3/3 AND finqa mean over the three >= +0.005 AND arena mean >= -0.002 on every checkpoint AND no subset <= -0.035 on any checkpoint AND `gold_full` >= banked - 0.005.

**KILL** - finqa negative anywhere, or mean < -0.005 on any checkpoint, or any subset <= -0.05.

Anything else is UNRESOLVED and is reported as such.

`gold_full` is a separate read on the private gold set and is not part of the arena dump; it is produced by the shipped gold read with the same correction applied, or reported as NOT MEASURED if that read cannot be run, in which case the ADMIT clause cannot be satisfied and the outcome is at best UNRESOLVED.

## 4. Explicitly forbidden after this point

- Changing the functional form (per-window offset, response-level offset, `log2`, `sqrt K`, `K - 1`, clipped or floored offsets, additive-in-characters variants)
- Re-fitting alpha on different data, a different model, a different stride grid, a different estimator, or a different sample size
- Sweeping alpha, or selecting among several alphas by any arena quantity
- Reporting a "best" alpha found after the dumps were reduced
- Substituting a different set of three checkpoints
