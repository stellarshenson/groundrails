# Acceptance criteria - groundrails grounder

The conditions the grounder must meet to ship, each paired with how it is verified. Automated criteria run in CI; the AWS end-to-end is local only.

## Correctness

- **Lexical accuracy** - macro-F1 ≥ 0.76 on the verified gold (leave-one-source-out GroupKFold) - verified by the lexical SOTA experiments ([`lexical-grounding-sota.md`](experiments/lexical-grounding-sota.md))
- **Semantic accuracy** - the `--semantic` cascade lifts macro-F1 to ≥ 0.82 on the same gold - verified by the semantic SOTA experiments ([`semantic-grounding-sota.md`](experiments/semantic-grounding-sota.md))
- **Determinism** - same input → same verdict, no sampling - frozen-weight design, covered by the suite

## Readiness gate

- **Library refuses before init** - `ground` / `ground_batch` / `grounding_document` raise `NotInitializedError` until `init()` runs - `tests/test_bootstrap.py::test_grounding_before_init_raises`
- **CLI refuses without groundrails.json** - `groundrails ground` exits 2 with an init hint when no `groundrails.json` is present - `::test_cli_ground_refuses_without_init`
- **init persists config** - `init` writes `groundrails.json` and never a `.stellars-plugins/settings.json` - `::test_init_writes_no_settings_json`

## Provisioning

- **3-way resolution** - each resource resolves override → S3 → local folder → HuggingFace - `tests/test_bootstrap.py` resolution tests
- **Provisioned calibration wins** - a provisioned calibration JSON overrides the bundled YAML block - `::test_provisioned_calibration_becomes_active`
- **Cross-lingual bridge** - the argos model auto-installs by default; missing + offline raises `UnsupportedLanguageError` rather than mis-scoring - CLI / grounding language tests

## CI gate (`.github/workflows/ci.yml`)

- **Lint** - `ruff check` + `ruff format --check` clean on `src/groundrails` and `tests`
- **Tests** - `pytest` green offline (`HF_HUB_OFFLINE=1`); model / integration tests skip cleanly
- **Build** - the wheel ships `config_document_processing.yaml`

## AWS end-to-end (local only, not CI)

- **Functional gate** - `aws/e2e.sh all` deploys the lexical grounder as a Lambda, invokes it, confirms the grounding result, and tears every resource down - needs AWS credentials, so it runs locally only ([`aws-deployment.md`](aws-deployment.md))
- **Expected result** - "The Eiffel Tower is in Paris." grounded; "The tower is 2000 metres tall." contradicted (`2000` vs `330`)
