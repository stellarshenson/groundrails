# Calibration howto

groundrails ships with frozen weights already fit on a 2,752-claim verified gold, so it works out of the box. Recalibrate only when your domain differs from the shipped gold - different document style, entity vocabulary, or language mix - and you have your own labelled claims to fit on. Calibration re-fits the frozen logistic weights; it never touches the deterministic layers.

There are two heads, fit independently:

- **Lexical manifolds** - the `low / medium / high` tiers; torch-free, the default path
- **Semantic joint head** - the `--semantic` switch; needs the OpenVINO cascade scores

## Prepare the dataset

One row per claim, each with the evidence it should be grounded against and a binary label.

- **Format** - a parquet file with columns `claim`, `source_text`, `label`, `lang`
- **`claim`** - the sentence to verify (string)
- **`source_text`** - the evidence passage the claim is checked against (string; the full source, not a pre-chunked list)
- **`label`** - `1` if the evidence supports the claim, `0` if it is a hallucination or contradiction
- **`lang`** - ISO code of the claim language (`en`, `de`, `no`, ...); leave empty to let the detector decide
- **Size** - a few hundred rows per class is the working minimum; the shipped gold uses 2,752 (786 hallucination / 1,966 supported)
- **Balance** - keep both classes present in meaningful numbers; a 5:1 skew biases the threshold

Place the file at `experiments/grounding-lexical/private-rag-forensics/gold/golden_grounding_evidence_verified.parquet`, or edit the `PRIVATE_RAG` path constant in the retrain script to point at yours.

## Run the calibration

### Lexical manifolds (low / medium / high)

```bash
cd experiments/grounding-lexical
uv run python retrain_manifolds.py
```

The script grounds every row through the shipped `high` lexical pipeline (a superset of all three tiers), fits one logistic per tier, and writes the result non-destructively into the trailing `calibration.lexical_manifolds` block of `src/groundrails/config_document_processing.yaml` - all comments and other config preserved. It prints each tier's feature count and threshold. No GPU; runtime scales with row count and source length (the parallel feature extraction is the cost).

### Semantic joint head (the `--semantic` switch)

```bash
cd experiments/grounding-semantic
uv run python joint_wirings.py
```

This races the three fusion wirings (escalation / always-both / reuse-seam) out-of-fold on the cached cascade scores, picks W1 escalation, folds the scaler into raw weights, and writes the frozen block to `reports/semantic_tier_block.yaml` plus the comparison to `reports/grounding_joint_wirings.md`. Paste the block into `calibration.semantic` in the config to ship it. The cascade scores must be cached locally first (per-pair reranker / NLI / cosine); reproducing them from scratch needs the int8 IRs and is the heavy step.

After either run, check the new weights with `groundrails config` (prints the resolved calibration block).

## What to expect

Recalibration helps most when your data is unlike the shipped gold; on data similar to it, expect little movement - the shipped weights are already near the optimum there.

- **Lexical baseline** - the shipped `high` manifold scores macro-F1 **0.817** on the private RAG gold; that is the bar a domain-matched retrain has to beat
- **Semantic switch** - turning on `--semantic` lifts macro-F1 from **0.759** (lexical-only) to **0.822** (+0.06) on the same gold, by recovering supported claims the conservative lexical manifold over-flags
- **Where the gain comes from** - the lexical head trades recall for determinism and over-flags supported claims (high false-positive rate); a domain-matched retrain narrows that, and the semantic head narrows it further
- **No free lunch** - too few rows, a heavy class skew, or evidence that does not actually contain the answer all produce weights worse than the shipped defaults; when in doubt, compare the retrained macro-F1 against 0.817 before shipping it
- **Determinism is preserved** - calibration only moves the frozen weights; inference stays a single logistic evaluation, no sampling, same input → same verdict
