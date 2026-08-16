# R20 fanout brief - pubmedqa absence-family hypotheses

Fable hypothesis-design agent (read-only, 2026-08-16). Input to the R20 fanout adjudication; the coordinator adjudicates. Design-pass statistics inside must be re-run and banked before citation.

All artifacts read, log greps complete, and fresh CPU probes run over the banked draw-2 and h159 dumps (which the H162 memo could not see — they had not landed when it ran). Brief follows.

---

# PubMedQA Absence-Family Attack — Research Brief (R19-H162 follow-on)

Repo: `/home/lab/workspace/private/ai-assistants/groundrails`. Canonical log = `docs/experiments/semantic-grounding-experiments.md` (cited as log :line). All new numbers below computed this session on CPU from banked artifacts; scripts inline, nothing trained, nothing touched on GPU.

## (a) Deepened failure analysis

**Standing.** pubmedqa across the four banked flagship draws (H150 d1/d2, H160 d3/d4): 0.5893 / 0.6298 / 0.6164 / 0.6636 → **k=4 subset mean 0.6248, empirical sd 0.0309** (computed from `experiments/grounding-semantic/R18-H150_arm_draw{1,2}_windowed_result.json`, `R19-H160_arm_draw{3,4}_windowed_result.json`). Faithful-oracle ceiling under the shipped read 0.7789 (`R12_label_ceiling_result.json`, via H162 memo :7); the whole 0.208 ceiling loss above that is cross-document conjunctive support — structural, not trainable (memo :72-75). Largest banked legal move: the R16-H142-T twin protocol's +0.066 (0.6725, memo :148). pubmedqa is 10% of the arena mean: subset +0.05 = mean +0.005.

**Failure anatomy** (R19-H162_pubmedqa_mechanisms.md, 250 items / 1,301 aligned sentences / 165 annotated-unsupported, h150d1):

- Sentence-level verifier works (0.6700 vs RAGBench sentence truth) but min-over-sentences destroys +0.0807 of it — the only large positive dilution in the arena (memo :26). Min is measured shut (R12: hard_min 0.7355 vs best alternative 0.7230; memo :74)
- Errors split exactly 44 FP / 44 FN at the stated threshold (memo :29)
- Family taxonomy (primary-class counts): **inference_not_stated 61 (37.0%)**, aim_vs_finding 26 (15.8%), relation_not_attested 12 (7.3%), scope_overextension 10, contradiction 10 (6.1%, the model's best family), entity_substitution 4, false_absence 1, unclassified 41 (24.8%, biased toward relation/entity per the memo's hand-read, :154)
- FN half signature: the sinking sentence of a positive item is its lowest-overlap sentence (containment 0.344 vs 0.536), scoring −3.93 — below the mean unsupported sentence (−3.16). Every banked near-miss lane builds negatives as HIGH-overlap minimal edits; nothing teaches that LOW overlap can still be support (memo :62-67)

**New this session — the memo's one-draw readings tested on the banked draw-2 and h159d1 dumps** (`R19-H161_pairs_h150d{1,2}.parquet`, `R19-H161_pairs_h159d1.parquet` joined to `R19-H162_pubmedqa_sentlabel.parquet`; fidelity: my d1 model AUROCs reproduce the memo bit-for-bit — 0.6700 overall, 0.6861/0.6463/0.6833/0.6338/0.6832 per family). Lexical baseline = max-over-windows `tok_containment` from the dump (harder than the memo's containment ratio; overall 0.6881):

| statistic | h150d1 | h150d2 | h159d1 | verdict |
|---|---|---|---|---|
| overall sent AUROC, model | 0.6700 | 0.7047 | 0.6549 | vs lexical 0.6881: below on d1/h159, above on d2 |
| inference_not_stated, model−lexical | **−0.0784** | **−0.0316** | **−0.1014** | deficit REPLICATES, same sign 3/3 |
| aim_vs_finding, model−lexical | +0.0086 | +0.0304 | +0.0240 | memo's "below word counting" is **baseline-dependent — does NOT replicate** under max-window containment; the family stands on mass + weak absolute separation (0.646-0.668) only |
| relation_not_attested, model−lexical | −0.0936 | −0.0252 | −0.0379 | replicates, but n=12 |
| corr(score, containment) among supported | 0.406 | 0.467 | 0.625 | paraphrase-coupling REPLICATES; worse in the prose-enriched h159 model |
| mean supported-item min-sentence score | −3.81 | −3.74 | −0.04 | prose enrichment (h159) lifted positive calibration wholesale yet pubmedqa item AUROC stayed ~flat |

Two honesty corrections this forces on the memo: (1) only **inference_not_stated** carries a replicated below-lexical deficit; aim_vs_finding's does not survive the baseline choice; (2) h159d1 (the mix containing PubHealth) read pubmedqa 0.6206 — the log's "+0.0111 watch-cell confirmation" (:3350s) was vs the 2-draw mean 0.6096; vs the k=4 mean 0.6248 it is **−0.004, i.e. the PubHealth transfer signal is NULL, not positive**.

## (b) Do-not-revisit list

1. **Learned window readout** — R16-H140: pubmedqa payload +0.0711 (seed-hardened +0.0741±0.0064, 4/4 same sign) at hotpotqa −0.043/tatqa −0.037; NOT LICENSED at bar; learned-readout route closed by the subset-blind serving shape (log :2723-2760, F1 correction, controls block)
2. **Count-aware / scalar aggregation** — R16-H141 DEAD pre-registration: pubmedqa has exactly 5.00 windows/sentence, zero dispersion; any count-aware correction is a per-subset constant, mathematically unable to move its AUROC; scalar ceiling = 13% of the payload, embedding branch double-controlled (log :2731 block + controls :2795)
3. **Trained aggregator** — R18-H156 KILLED at draw 1 (training cost −0.0250, serving-neutral); resolves-below under Amendment V1 re-adjudication (log :3294, V1 block)
4. **Serving-read amendments generally** — R18-H151: pooling class closed, max is best-mean AND minimum-spread (log :3281-3288); R12 precursor P-B (discourse-marker sentence exclusion) killed, oracle bound +0.0065 (memo :48); the brief-B serving-convention door (question conditioning + joint pool) is ESCALATED TO THE AUTHOR, not actionable (log :3861); H165 concatenated-pool killed globally — tables pay (log :3553-3572). **Per the assignment: no serving-read proposals below**
5. **Wikipedia-register bulk mass** — drove pubmedqa 0.5665 → 0.4783 below chance (R8-H83, log :1447-1457); VitaminC near-miss fix recovered only to 0.5466 (:1473-1483); standing warning in memo :118
6. **Bulk enriched mix** — R19-H159 KILLED (−0.026 mean, tables collapsed); future corpus use is per-corpus, register-matched only (log :3347-3355)
7. **Batch co-location** — R14-H135 killed; pubmedqa fell to 0.5256; family closed absent an author-ordered variant (log :2749 block)
8. **Length / sentence-count heuristics** — 0.6347 beats the model but is selection on the benchmark; forbidden (memo :28, :157)
9. **scope_bind lane** — under-evidenced at n=17, recorded not recommended (memo :121-128)
10. **Cross-document conjunctive support** — the 0.208 ceiling slice; max-over-windows is an OR; not a lane target (memo :72-75)
11. **PubMedQA-derived data** — contamination wall; R12 ruling 8 admits the biomedical register only via SciFact/HealthVer-class corpora, "NOT PubMedQA itself" (log :2488)
12. **Not-yet-acted nominations confirmed open**: the H162 pubmedqa nominations are recorded (log :3447-3458, build-first assert_vs_infer) but **no arm is registered**; R20 ruling 4 licensed hagrid/emanual (H174) and the three-way objective (H166-A1) ahead of them — any pubmedqa arm queues behind H174 unless the author reorders. H171's caution stands: weakest ≠ most improvable (log :3755)

## (c) Hypotheses, registration shape, ranked

Shared arithmetic, stated once: pooled per-draw sd is frozen at 0.01090 (Amendment V1); mean floors k=2 0.01542 / k=3 0.01259 / k=4 0.01090. Every hypothesis below predicts a MEAN effect of at most +0.007 — **sub-floor at any affordable k** — so per the adopted protocol each PRIMARY is a mechanism gate on non-arena data, the arm is exploratory on the mean, and the arena carries only a guard plus a pubmedqa subset-level reading. At the subset level pubmedqa's seed sd is 0.0216-0.031; 2×SE at k=4 ≈ 0.022-0.031, so a subset effect ≥ +0.03 is resolvable at k=4 and nothing smaller is. One draw ≈ 6.5 GPU-h.

### PM-1 — assert_vs_infer evidence-deletion contrast lane (rank 1; adopt memo's build-first, with a construction upgrade)

- **Claim**: because inference_not_stated is 37% of pubmedqa's unsupported sentences and the flagship reads it below a max-window lexical-containment baseline on both banked draws (−0.0784 d1, −0.0316 d2, −0.1014 h159d1), a ~12-20k-row contrast lane whose negatives REMOVE the stating sentence from the evidence — positives = claim + evidence minus one random non-rationale sentence, negatives = claim + the same evidence minus the rationale sentence, so the pair is length- and structure-matched by construction and only sentence IDENTITY differs — added as one DANN group to the flagship recipe will lift a held-out source-disjoint deletion-probe AUROC to ≥ 0.75 (control predicted 0.55-0.65) and pubmedqa k=4 mean by +0.02..+0.05, while the H159 table guard (finqa/tatqa/delucionqa within one across-seed spread of flagship subset means) and gold_full ≥ 0.84 hold
- **Diagnostic kill-gate (pre-build, CPU, banked)**: (i) deficit replication across both flagship draws — **RUN THIS SESSION, PASS** (same sign 3/3, table above); (ii) supply census: ≥ 8k (claim, localizable rationale, multi-sentence evidence) triples extractable from MiniCheck C2D/D2C + FAVA spans without SciFact — below 8k the lane waits on the SciFact licence or dies; (iii) built-lane leak suite (H133-v2 pattern): claim-only TF-IDF < 0.55 (trivially near-chance — the claim is identical across the pair), evidence-length-delta AUROC in [0.45,0.55] (guaranteed by the matched-deletion construction), R14-H136 8-gram census re-run on the built pairings
- **Supply**: MiniCheck C2D/D2C 14,395 rows MIT (gate 0.0); FAVA 30,073 CC-BY-4.0 (gate 0.000116); SciFact 508 SUPPORT rows with rationale keys, admitted 2026-08-09 on upstream CC BY 4.0 + ODC-By (NC discrepancy recorded — MiniCheck+FAVA can carry the lane alone if it stays unresolved). All gate-green vs the walled corpora; none PubMedQA-derived. Novel in kind: every banked lane corrupts the CLAIM; this one removes the EVIDENCE
- **Predicted**: pubmedqa +0.02..+0.05 (attenuated by the +0.0807 min-dilution); mean +0.002..+0.005 — sub-floor, exploratory; declared k=2 (extend to k=4 only if draw 1 subset read ≥ +0.03)
- **Cost**: ~1 day CPU build; 13 GPU-h at k=2
- **Kills it**: deletion probe arm ≤ control + 0.05; any table-guard breach; k=2 arena mean < flagship k=6 mean − 0.01542

### PM-2 — paraphrase-support positive-side lane (rank 2)

- **Claim**: because the FN half of the error budget (44/88) is set by supported low-overlap sentences (min-sentence score −3.81/−3.74 across draws, below the unsupported mean −3.16) and score-containment coupling among supported sentences replicates (r = 0.41/0.47), a ~10-15k-row positive-side lane of low-containment SUPPORTED pairs — FActScore (13,653 rows MIT, atomic facts are abstractive restatements by construction) + AttributionBench (16,444 rows Apache-2.0, ExpertQA/HAGRID carved out) filtered to the low-containment supported tail, PLUS the banked judge-certified paraphrase label-1 band from R10-H111/DR (2,982 + 527 rows, recorded as a free augmentation lever and never used, log :2265) — with containment-matched unsupported negatives so within-lane containment-label AUROC sits in [0.45,0.55], will reduce held-out corr(score,containment | supported) on non-arena data (RAGTruth EN held-out) by ≥ 0.15 and lift pubmedqa without raising the FP rate (anti-gaming diagnostic and gold_full guards green)
- **Diagnostic kill-gate (pre-build, CPU, banked)**: (i) coupling replication — **RUN, PASS**; (ii) supply census: ≥ 8k supported rows at containment ≤ 0.3 across the two corpora plus the H111 band; (iii) the containment-decorrelation bar on the built lane ([0.45,0.55]) — a lane that fails it would teach "low overlap ⇒ fine", the exact FP hazard
- **Evidence-against, recorded**: h159d1 (prose-enriched, confounded across 5 lanes) lifted positive calibration wholesale (sup-min −3.8 → −0.04) yet pubmedqa stayed ~flat and its low-containment discrimination worsened — a calibration shift is not a ranking fix. The decorrelation bar is the designed answer; if the mechanism gate passes and pubmedqa still does not move, that reading was right and the family closes
- **Predicted**: pubmedqa +0.015..+0.04; mean +0.002..+0.004 — sub-floor, exploratory, mechanism PRIMARY; k=2
- **Cost**: supply fully banked; ~0.5 day CPU; 13 GPU-h
- **Kills it**: coupling-drop < 0.15; FP-side guard breach; table guard

### PM-3 — PubHealth solo register-matched lane (rank 3)

- **Claim**: because pubmedqa's failures are 87% absence-type and PubHealth's unproven/mixture verdicts are naturally-occurring absence-type negatives in the health register (12,251 usable rows, MIT, gate green, `data/external/datasets/dataset-pubhealth.md`), admitting PubHealth ALONE as one DANN group — removing the H159 confound whose four sibling lanes are diagnosed as the table-collapse cause (log :3355) — will lift pubmedqa while the table guard holds
- **Honesty**: the only prior is null — h159d1 pubmedqa 0.6206 is −0.004 vs the k=4 flagship mean; the arm's case is confound removal plus register match, not a measured positive. This is exactly the "single-corpus ablation, cheaper 2-arm version" the H159 verdict recorded as the author's decision, aimed at pubmedqa instead of hagrid
- **Diagnostic kill-gate (pre-build, CPU, banked)**: on the banked h159d1 pubmedqa dump, the unproven/mixture-style absence families must NOT be the families h159 made worse — measured: inference_not_stated margin worsened to −0.1014 under h159. That gate is **MARGINAL-FAIL as evidence of transfer**; I recommend registering PM-3 only as a rider ablation behind PM-1/PM-2, not as a standalone arm
- **Predicted**: pubmedqa 0..+0.03; mean ≤ +0.003 — exploratory; k=2. Cost: zero acquisition, 13 GPU-h. **Kills it**: pubmedqa draw-1 below flagship subset mean; table guard

### PM-4 — aim_vs_finding lane from ClinicalTrials.gov posted results (rank 4)

- **Claim**: because 15.8% of unsupported sentences are aims credited as findings and the corpus offers aims three times as often as results (33.7% vs 13.0% of evidence documents), a ~10-15k-pair lane of registry minimal pairs — claim = a trial's posted result; evidence-pos = the posted result field; evidence-neg = the same trial's pre-stated primary outcome measure (identical content words, different evidential status; the containment baseline is at chance by construction) — will read held-out aim-vs-result probe AUROC ≥ 0.80 vs a near-chance flagship baseline leg
- **Honesty correction to the memo**: the family's below-lexical margin does NOT replicate under the max-window baseline (+0.0086/+0.0304/+0.0240); the case rests on mass and weak absolute separation (0.646-0.668, 3/3 reads below each draw's overall), so the predicted subset effect shrinks to +0.01..+0.03 — likely unresolvable even at the subset level at k=4
- **Diagnostic kill-gate (pre-build, CPU)**: (i) family separation < overall on both banked draws — **RUN, PASS (weakly)**; (ii) AACT supply census after fetch: ≥ 8k completed trials with both fields non-degenerate; (iii) full acquisition path per dataset rules (fetch CLI + sidecar + H136 gate) BEFORE any lane build — the only hypothesis here needing new acquisition
- **Supply**: ClinicalTrials.gov / CTTI AACT, US-government public domain (re-verify at pull); contamination clear by construction (registry ≠ PubMed abstracts), H136 census binding
- **Predicted**: mean ≤ +0.003, exploratory, mechanism PRIMARY (the crispest probe of the five); k=2. Cost: ~2 days CPU acquisition+build, 13 GPU-h. **Kills it**: probe baseline leg already ≥ 0.70 on the flagship (skill exists, lane redundant); probe arm < 0.80; table guard

### PM-5 — NEI-channel objective lever on the recovered three-way labels (rank 5); VitaminC-NEI mining REFUTED

- Candidate (d) as posed — hard-negative mining within VitaminC NEI rows — is **refuted on the record**: Wikipedia-register mass measurably harmed pubmedqa (0.5665 → 0.4783, recovery only 0.5466; log :1447-1483), the memo carries the same warning for this exact family (:118), and H159 re-taught the composition lesson. Do not build a Wikipedia-register absence lane for a biomedical deficit
- The salvageable objective lever: R19-H166-A1 recovered and validated 400,653 three-way labels and wired a +769-param aux `con_head` for CONTRADICTED (log :3880 block). The absence-side analogue — an `nei_head` trained MIL-max BCE on 1[y3 == NEI], task_head untouched — targets the 87% absence mass with the identical exploratory framing, near-zero marginal cost if ridden as an H166 amendment (same draws, one more masked BCE term, lambda fixed)
- **Diagnostic kill-gate**: none computable on banked artifacts — the falsifiable content is the same contrast H166-A1 uses (held-out VitaminC NEI-vs-SUPPORTED AUROC on the new channel vs a near-chance flagship baseline leg). PRIMARY = that mechanism gate ≥ 0.85; arena strictly exploratory (predicted mean [−0.005, +0.008], the H166 numbers)
- **Kills it**: mechanism gate fails → this trunk/objective cannot carry the distinction and the absence-channel family closes; register-transfer risk (labels are Wikipedia-register) is priced by the pubmedqa subset guard

### Recommended registration shape

Bundle PM-1 + PM-2 as a two-lane portfolio arm on the H174 pattern (separate DANN groups, per-lane mechanism-gate PRIMARIES, one training arm): predicted pubmedqa +0.035..+0.08 → mean +0.004..+0.008 — still sub-floor on the mean, but the joint subset effect crosses the pubmedqa k=4 subset resolvability line (~+0.03), giving the arm one honest arena-resolvable secondary alongside two falsifiable mechanism gates, at 26 GPU-h for k=4. PM-3 rides only as an optional third group; PM-4 queues behind its acquisition; PM-5 rides on H166. Sequencing per R20 ruling 4: behind H172 (in flight) and H174 unless the author reorders.

**Key artifact paths**: memo `/home/lab/workspace/private/ai-assistants/groundrails/experiments/grounding-semantic/R19-H162_pubmedqa_mechanisms.md`; banked dumps used for the replication probes `/home/lab/workspace/private/ai-assistants/groundrails/experiments/grounding-semantic/R19-H161_pairs_h150d{1,2}.parquet`, `R19-H161_pairs_h159d1.parquet`, `R19-H162_pubmedqa_sentlabel.parquet`; supply cards under `/home/lab/workspace/private/ai-assistants/groundrails/data/external/datasets/` (`dataset-minicheck.md`, `dataset-fava.md`, `dataset-scifact.md`, `dataset-factscore.md`, `dataset-attributionbench.md`, `dataset-pubhealth.md`); canonical log `/home/lab/workspace/private/ai-assistants/groundrails/docs/experiments/semantic-grounding-experiments.md`.
