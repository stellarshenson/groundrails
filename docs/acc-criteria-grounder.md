# Acceptance criteria - groundrails grounder

The conditions the grounder must meet to ship, each paired with how it is verified. Automated criteria run in CI; the AWS end-to-end is local only.

## Authors

- `@kj` Konrad Jelen

## Correctness `CORR`

accuracy and repeatability of the verdicts

- [x] `ACC-CORR-1` **Lexical accuracy** - CRITICAL; macro-F1 ≥ 0.76 on the verified gold (leave-one-source-out GroupKFold) - verified by the lexical SOTA experiments ([`lexical-grounding-sota.md`](experiments/lexical-grounding-sota.md))
  - evidence: lexical SOTA Performance table: shipped head macro-F1 0.817 on the private RAG gold, grouped CV (bar 0.76)
  - test-tags: MANUAL
  - test: fit the shipped lexical head with grouped CV on the private RAG gold, read macro-F1
  - log: 2026-09-25T02:15:30Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:31Z @kj edited importance added "CRITICAL"; test added "fit the shipped lexical head with grouped CV on the private RAG gold, read macro-F1"; test-tags added "MANUAL"
  - log: 2026-09-25T02:15:45Z @kj closed: verified at `d671df2` plus working tree
- [ ] `ACC-CORR-2` **Semantic accuracy** - HIGH; the `--semantic` cascade lifts macro-F1 to ≥ 0.82 on the same gold - verified by the semantic SOTA experiments ([`semantic-grounding-sota.md`](experiments/semantic-grounding-sota.md))
  - test-tags: MANUAL
  - test: run the semantic cascade end-to-end on the private gold, read macro-F1
  - log: 2026-09-25T02:15:30Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:31Z @kj edited importance added "HIGH"; test added "run the semantic cascade end-to-end on the private gold, read macro-F1"; test-tags added "MANUAL"
  - log: 2026-09-25T02:15:46Z @kj not met: semantic SOTA records the shipped cascade at macro-F1 0.789 end-to-end, 0.796 out-of-fold; 0.824 belonged to the retired 6-model plus lexical stack
- [x] `ACC-CORR-3` **Determinism** - CRITICAL; same input -> same verdict, no sampling - frozen-weight design, covered by the suite
  - evidence: `test_threaded_batch_matches_serial` and `test_grounding_matches_parent_golden` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: FUNCTIONAL
  - test: `pytest tests/test_functional_pipeline.py -k test_threaded_batch_matches_serial`
  - log: 2026-09-25T02:15:30Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:31Z @kj edited importance added "CRITICAL"; test added "`pytest tests/test_functional_pipeline.py -k test_threaded_batch_matches_serial`"; test-tags added "FUNCTIONAL"
  - log: 2026-09-25T02:15:45Z @kj closed: verified at `d671df2` plus working tree

## Readiness gate `READY`

refusing to ground before `init()` has run

- [x] `ACC-READY-4` **Library refuses before init** - HIGH; `ground` / `ground_batch` / `grounding_document` raise `NotInitializedError` until `init()` runs - `tests/test_bootstrap.py::test_grounding_before_init_raises`
  - evidence: `test_grounding_before_init_raises` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT
  - test: `pytest tests/test_bootstrap.py::test_grounding_before_init_raises`
  - log: 2026-09-25T02:15:30Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:31Z @kj edited importance added "HIGH"; test added "`pytest tests/test_bootstrap.py::test_grounding_before_init_raises`"; test-tags added "UNIT"
  - log: 2026-09-25T02:15:45Z @kj closed: verified at `d671df2` plus working tree
- [x] `ACC-READY-5` **CLI refuses without groundrails.json** - HIGH; `groundrails ground` exits 2 with an init hint when no `groundrails.json` is present - `::test_cli_ground_refuses_without_init`
  - evidence: `test_cli_ground_refuses_without_init` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT
  - test: `pytest tests/test_bootstrap.py::test_cli_ground_refuses_without_init`
  - log: 2026-09-25T02:15:30Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:31Z @kj edited importance added "HIGH"; test added "`pytest tests/test_bootstrap.py::test_cli_ground_refuses_without_init`"; test-tags added "UNIT"
  - log: 2026-09-25T02:15:45Z @kj closed: verified at `d671df2` plus working tree
- [x] `ACC-READY-6` **init persists config** - MEDIUM; `init` writes `groundrails.json` and never a `.stellars-plugins/settings.json` - `::test_init_writes_no_settings_json`
  - evidence: `test_init_writes_no_settings_json` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT
  - test: `pytest tests/test_bootstrap.py::test_init_writes_no_settings_json`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:31Z @kj edited importance added "MEDIUM"; test added "`pytest tests/test_bootstrap.py::test_init_writes_no_settings_json`"; test-tags added "UNIT"
  - log: 2026-09-25T02:15:45Z @kj closed: verified at `d671df2` plus working tree

## Provisioning `PROV`

where models and calibration are loaded from

- [x] `ACC-PROV-7` **3-way resolution** - HIGH; each resource resolves override -> S3 -> local folder -> HuggingFace - `tests/test_bootstrap.py` resolution tests
  - evidence: the 4 `test_resolution_*` tests in `tests/test_bootstrap.py` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT
  - test: `pytest tests/test_bootstrap.py -k resolution`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "HIGH"; test added "`pytest tests/test_bootstrap.py -k resolution`"; test-tags added "UNIT"
  - log: 2026-09-25T02:15:46Z @kj closed: verified at `d671df2` plus working tree
- [x] `ACC-PROV-8` **Provisioned calibration wins** - MEDIUM; a provisioned calibration JSON overrides the bundled YAML block - `::test_provisioned_calibration_becomes_active`
  - evidence: `test_provisioned_calibration_becomes_active` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT
  - test: `pytest tests/test_bootstrap.py::test_provisioned_calibration_becomes_active`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "MEDIUM"; test added "`pytest tests/test_bootstrap.py::test_provisioned_calibration_becomes_active`"; test-tags added "UNIT"
  - log: 2026-09-25T02:15:46Z @kj closed: verified at `d671df2` plus working tree
- [x] `ACC-PROV-9` **Cross-lingual bridge** - HIGH; the argos model auto-installs by default; missing + offline raises `UnsupportedLanguageError` rather than mis-scoring - CLI / grounding language tests
  - evidence: `tests/test_unsupported_language.py` and `test_ground_cross_lingual_supported` green; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT, FUNCTIONAL
  - test: `pytest tests/test_unsupported_language.py tests/test_cli.py::test_ground_cross_lingual_supported`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "HIGH"; test added "`pytest tests/test_unsupported_language.py tests/test_cli.py::test_ground_cross_lingual_supported`"; test-tags added "UNIT, FUNCTIONAL"
  - log: 2026-09-25T02:15:46Z @kj closed: verified at `d671df2` plus working tree

## CI gate (`.github/workflows/ci.yml`) `CI`

the GitHub Actions lint, test and build checks

- [x] `ACC-CI-10` **Lint** - MEDIUM; `ruff check` + `ruff format --check` clean on `src/groundrails` and `tests`
  - evidence: `uvx ruff check`: All checks passed; `ruff format --check`: 38 files already formatted, on 2026-09-25
  - test-tags: CI
  - test: `uvx ruff check src/groundrails tests` and `uvx ruff format --check src/groundrails tests`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "MEDIUM"; test added "`uvx ruff check src/groundrails tests` and `uvx ruff format --check src/groundrails tests`"; test-tags added "CI"
  - log: 2026-09-25T02:15:46Z @kj closed: verified at `d671df2` plus working tree
- [x] `ACC-CI-11` **Tests** - HIGH; `pytest` green offline (`HF_HUB_OFFLINE=1`); model / integration tests skip cleanly
  - evidence: HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - test-tags: UNIT, FUNCTIONAL
  - test: `HF_HUB_OFFLINE=1 uv run pytest tests -q`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "HIGH"; test added "`HF_HUB_OFFLINE=1 uv run pytest tests -q`"; test-tags added "UNIT, FUNCTIONAL"
  - log: 2026-09-25T02:15:46Z @kj closed: verified at `d671df2` plus working tree
- [x] `ACC-CI-12` **Build** - CRITICAL; the wheel ships `config_document_processing.yaml`
  - evidence: `uv build --wheel` 1.1.1: wheel lists `groundrails/config_document_processing.yaml` (17040 bytes), 2026-09-25
  - test-tags: CI
  - test: `uv build --wheel`, list the wheel, find `groundrails/config_document_processing.yaml`
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "CRITICAL"; test added "`uv build --wheel`, list the wheel, find `groundrails/config_document_processing.yaml`"; test-tags added "CI"
  - log: 2026-09-25T02:15:46Z @kj closed: verified at `d671df2` plus working tree

## AWS end-to-end (local only, not CI) `AWS`

the Lambda deployment test, run by hand with AWS credentials

- [ ] `ACC-AWS-13` **Functional gate** - MEDIUM; `aws/e2e.sh all` deploys the lexical grounder as a Lambda, invokes it, confirms the grounding result, and tears every resource down - needs AWS credentials, so it runs locally only ([`aws-deployment.md`](aws-deployment.md))
  - test-tags: E2E
  - test: `aws/e2e.sh all` with AWS credentials
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "MEDIUM"; test added "`aws/e2e.sh all` with AWS credentials"; test-tags added "E2E"
  - log: 2026-09-25T02:15:46Z @kj no recorded run found in the journal or logs; stays open until `aws/e2e.sh all` is run
- [ ] `ACC-AWS-14` **Expected result** - MEDIUM; "The Eiffel Tower is in Paris." grounded; "The tower is 2000 metres tall." contradicted (`2000` vs `330`)
  - test-tags: E2E
  - test: `aws/e2e.sh invoke`, read the two verdicts
  - log: 2026-09-25T02:15:31Z @kj imported from the legacy criteria list
  - log: 2026-09-25T02:15:32Z @kj edited importance added "MEDIUM"; test added "`aws/e2e.sh invoke`, read the two verdicts"; test-tags added "E2E"

## Claim extraction `CLAIM`

turning a document into the claims that get grounded

- [x] `ACC-CLAIM-15` **Non-English document warns** - HIGH; a document whose prose is confidently not English logs one warning naming the language and the multilingual bridge; English and undetermined text log nothing
  - evidence: `TestNonEnglishWarning` 6 green; warning fired on all 7 documents of a Polish field batch; HF_HUB_OFFLINE=1 pytest 428 passed, 4 skipped on 2026-09-25
  - related: DEF-CLAIM-26 - the defect this criterion closes
  - test: `pytest tests/test_defect_regressions.py -k NonEnglish`
  - test-tags: UNIT
  - log: 2026-09-25T02:15:46Z @kj added
  - log: 2026-09-25T02:15:46Z @kj closed: implemented: `warn_if_not_english` in `extract_claims`

