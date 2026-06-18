# Semantic grounder (research leg)

The model-based counterpart to the deterministic lexical grounder in `../grounding-lexical/`. Classifies each claim as supported or hallucination with two cross-encoders (relevance reranker + NLI entailment) over a bi-encoder pre-filter, combined by a small logistic - no fine-tuning. Imported from the source datascience project; works independently and is **not yet integrated** with the lexical grounder.

## Pipeline

- **Pre-filter** - `bge-m3` bi-encoder ranks evidence chunks, keeps top-k per claim
- **Score** - `bge-reranker-v2-m3` (relevance) + `mDeBERTa-v3-nli` (entailment) score the survivors, max over chunks
- **Classify** - logistic over the max-over-chunks scores → grounded probability
- **Serving** - all three models run as **OpenVINO int8 on a single runtime**, torch-free; published IRs pulled from HuggingFace (`stellars/*-openvino-int8`)

## Modules

- **`grounding_openvino.py`** - the deployable serving core: IR load/compile, embed / rerank / NLI scoring, cascade + cosine-gate + early-exit skip mechanisms
- **`grounding_models.py`** - gold loader (`load_gold`, `chunk_text`), metrics, and the GPU/torch reference scoring that built the cached per-model `.npy` (torch imported lazily)
- **`grounding_ensemble.py`** - the meta-classifier over the six cached model scores + a lexical contradiction flag; the headline macro-F1 / AUC report
- **`grounding_hypotheses.py`** - the H9-H14 hypothesis rounds (cascade band, cosine gate, early-exit, fused evidence)

Scripts: `score_ov_pipeline.py` / `score_int8_fullgold.py` (cache the int8 pipeline scores), `run_grounder_full.py` (full end-to-end serving run + latency), `bench_grounder_*.py` (latency benchmarks), `build_ov_grounder.py` / `push_ov_models_to_hf.py` (build/publish the int8 IRs).

## Data

The gold + caches live under `private-rag-forensics/` (gitignored - private, never committed):

- `gold/golden_grounding_evidence_verified.parquet` - 2,752 dual-judge labelled claims (`{claim, source_text, label, lang}`); recipe in `../../docs/dataset/2026-06-01-how-to-build-golden-dataset.md`
- `model_scores/*.npy` + `ensemble_features.npz` - cached per-model scores so the ensemble reproduces without a GPU
- `grounding_probe_questions.json`, `grounding_eval.json` - probe set + eval inputs

## Run

Scripts import siblings flat - run them by path (Python puts the script's dir on `sys.path`):

```bash
# deterministic meta-classifier eval (CPU, seconds, cached scores - no models needed)
python experiments/grounding-semantic/grounding_ensemble.py     # → reports/grounding_ensemble.md

# full int8 serving run (pulls ~1.4 GB of IRs from HF; smoke = first N claims)
python experiments/grounding-semantic/run_grounder_full.py 12
```

Deps: `uv sync --extra semantic-grounder` (transformers + polars; openvino / huggingface-hub / scikit-learn already core). torch is dev-only - the cached-score eval and the int8 serving path are torch-free.

## Equivalence

`tests/test_semantic_grounder_equivalence.py` asserts the ported ensemble reproduces the parent byte-for-byte (skips when the gitignored data is absent):

- meta-classifier (gbm) OOF AUC **0.913**, macro-F1 **0.824**, FP/FN **248 / 160**
- best single signal (bge-reranker) macro-F1 **0.757**
