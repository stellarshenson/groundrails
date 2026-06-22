# Calibration reference

groundrails ships frozen logistic weights fit on a verified gold set, so it grounds correctly with no setup. Recalibration re-fits those weights to your own labelled claims; it never touches the deterministic recall layers, and inference stays a single logistic evaluation - same input → same verdict.

## When to recalibrate

- **Domain drift** - your documents, entity vocabulary, or language mix differ from the shipped gold
- **You have labels** - a few hundred labelled claims per class, minimum
- **Not otherwise** - on data like the shipped gold the frozen weights are already near optimal; a thin or skewed retrain ships worse weights

## Two heads, fit independently

- **Lexical manifolds** - the `low / medium / high` tiers; torch-free, the default path
- **Semantic joint head** - the `--semantic` switch; needs the OpenVINO cascade scores

## Dataset

- **Format** - a parquet file with columns `claim`, `source_text`, `label`, `lang`
- **`claim`** - the sentence to verify
- **`source_text`** - the full evidence passage the claim is checked against, not a pre-chunked list
- **`label`** - `1` supported, `0` hallucination or contradiction
- **`lang`** - ISO code (`en`, `de`, `no`, …); leave empty to let the detector decide
- **Size** - a few hundred rows per class is the working minimum; the shipped gold uses 2,752 (786 hallucination / 1,966 supported)
- **Balance** - keep both classes present; a 5:1 skew biases the threshold
- **Path** - `experiments/grounding-lexical/private-rag-forensics/gold/golden_grounding_evidence_verified.parquet`, or repoint the `PRIVATE_RAG` constant in the retrain script

## Fit

- **Lexical** - `cd experiments/grounding-lexical && uv run python retrain_manifolds.py` - grounds every row through the `high` pipeline (a superset of all three tiers), fits one logistic per tier, writes `calibration.lexical_manifolds` into `src/groundrails/config_document_processing.yaml` with comments preserved; prints each tier's feature count and threshold; no GPU
- **Semantic** - `cd experiments/grounding-semantic && uv run python joint_wirings.py` - races the three fusion wirings out-of-fold on cached cascade scores, picks W1 escalation, writes the frozen block to `reports/semantic_tier_block.yaml`; paste it into `calibration.semantic`; the cascade scores must be cached first (reproducing them needs the int8 IRs)
- **Check** - `groundrails config` prints the resolved calibration block

## Export the provisioned JSON

- **`groundrails calibration export -o calibration.json`** - writes the active calibration block to a standalone JSON, the artifact a deployment provisions
- **`--source <override>`** - export from a specific calibration source instead of the active one
- **Python** - `from groundrails import export_calibration; export_calibration("calibration.json")`

## How a deployment loads it

- **`groundrails init --calibration s3://…/calibration.json`** - provisions it at startup (an `s3://…`, `https://…`, or local path)
- **`GROUNDRAILS_CALIBRATION_JSON=/path/calibration.json`** - points the runtime at it directly
- **Precedence** - a provisioned JSON wins over the bundled YAML calibration block
- **`$GROUNDRAILS_HOME`** - `init` writes `groundrails.json` (the runtime config) here, or `./groundrails.json` when it is unset; a later `groundrails ground` reads it from the same place

## What to expect

- **Lexical baseline** - the shipped `high` manifold scores macro-F1 **0.817** on the private RAG gold; the bar a domain retrain has to beat
- **Semantic switch** - `--semantic` lifts macro-F1 from **0.759** to **0.822** (+0.06) by recovering supported claims the conservative lexical manifold over-flags
- **No free lunch** - too few rows, heavy class skew, or evidence that lacks the answer all produce weights worse than the shipped defaults; compare against 0.817 before shipping
- **Determinism preserved** - calibration moves the frozen weights only; inference stays deterministic
