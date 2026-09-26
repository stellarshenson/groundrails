# Changelog

Notable changes to the `groundrails` package, one section per PyPI release, newest first. Version numbers that were set but never published (1.0.36, 1.0.37, 1.1.0) are folded into the next published release. Defect ids refer to [docs/defects.md](docs/defects.md).

## [1.1.2] - 2026-09-26

### Added

- `extract_claims` logs a warning when the document is not English, because extraction without the multilingual bridge keeps almost no sentences of other languages (DEF-CLAIM-26)
- `ground`, `ground_batch` and `grounding_document` accept sources as a `{path: text}` mapping; before, every claim was grounded against the file names instead of the text (DEF-GROUND-21)

### Changed

- A sentence opening with `if` or `unless` is no longer marked `hypothetical` by `out_of_scope`, so sourced rules such as legal or clinical conditionals are grounded in full; an `either ..., or ...` sentence without digits is still marked (DEF-CLAIM-29, DEF-CLAIM-36)

### Fixed

- A value the source states verbatim is never returned as CONTRADICTED (DEF-NUMBER-22)
- A citation marker such as `[1]` is not read as a stated number, in grounding and in `check_consistency` (DEF-NUMBER-30, DEF-SELF-31)
- A range (`5-12 years`, `06:00-09:00`, `between 5 and 12`) is one value in `check_consistency`, not two divergent ones (DEF-SELF-17)
- A range end keeps its minus sign, so `-5 to 10 degrees` stated twice is consistent (DEF-NUMBER-48)
- A date after `before` or `after` is read as a bound, not as an exact value (DEF-NUMBER-42)
- The comparative words (`over`, `under`, `up to` and the rest) match only as whole words, so `cover 5 million` no longer hides a wrong number (DEF-NUMBER-47)
- A number of 29 or more digits, or above the float range, no longer makes the numeric check raise (DEF-NUMBER-43, DEF-NUMBER-49)
- HTML comments are removed before extraction, only outside fenced and inline code, with LF or CRLF line ends, and line numbers are kept (DEF-CLAIM-27, DEF-CLAIM-35)
- Consecutive list items, also inside nested blockquotes, stay separate claims (DEF-CLAIM-24, DEF-CLAIM-38)
- A sentence ends after a closing quote or bracket, and a citation marker after the full stop stays with the sentence it cites (DEF-CLAIM-28, DEF-CLAIM-41)

## [1.1.1] - 2026-09-25

### Added

- `groundrails.dataset` and the `groundrails dataset` command: a corpus preprocessing pipeline for grounding training data in six stages (fetch, contaminate, format, shape, assemble, census), with corpora declared in `corpora.yaml`; needs the `[dataset]` extra
- `out_of_scope()` marks hypotheticals, document self-references and directives, recorded on `ExtractedClaim.out_of_scope_reason` and `GroundingMatch.out_of_scope_reason`; with the semantic tier on, such a claim keeps its lexical verdict and skips the cascade (DEF-PERF-20)
- `ComponentNotReadyError`: `ground` and `ground_batch` raise it before any claim when the semantic tier is on and its extra is not installed (DEF-SEMA-15)
- A one-time warning when a Hugging Face model download starts without a token, since anonymous downloads are rate-limited

### Changed

- Python requirement relaxed from `~=3.12.0` to `>=3.11` (DEF-PACK-18)
- `polars` is a core dependency

### Fixed

- A semantic cascade failure on one claim keeps the lexical verdict already computed instead of returning a blank match (DEF-SEMA-14)
- A batch in which every claim errors raises instead of returning results that read as 0% grounded (DEF-SEMA-16)

## [1.0.38] - 2026-07-22

### Added

- `groundrails.calibrate()` and `groundrails calibration fit`: re-fit the verdict calibration from labelled `(claim, source_text, label)` records and write it as a config block
- `groundrails calibration eval`: score a calibration against labelled records (macro-F1 and per-class precision, recall and F1)

### Fixed

- The sentence splitter no longer splits inside citations such as `et al. 2008` and `Buchanan, C. (1991)` (DEF-CLAIM-1)
- Content under reference headings such as `## Sources` is not extracted as claims (DEF-CLAIM-2)
- The claim filter rejects fragments and contentless references such as `This is ...` (DEF-CLAIM-3)
- Claims with an invariant past-tense verb (`hit`, `cut`, `set`, `cost`) are extracted
- Short claims no longer ground on BM25 token recall alone: `bm25_min_claim_tokens` sets a floor (DEF-GROUND-4)
- A claim present verbatim in the evidence is grounded even when the trained head scores it low; contradiction checks still apply (DEF-GROUND-9)
- Fuzzy evidence quotes expand to whole words (DEF-GROUND-6)
- Numeric context keys keep the sign and the nearest content word, so unrelated percentages no longer collide into a false CONTRADICTED (DEF-NUMBER-5)
- `verification_needed` no longer fires on every verbatim numeric match, and it flags an NLI contradiction score just below the floor (DEF-GROUND-10)
- CLI threshold flags are read, and their defaults no longer override the YAML config (DEF-GROUND-11)
- `--semantic` without a value no longer overrides `calibration.mode: semantic` from the config
- The CLI exit code counts a CONTRADICTED claim as a failure (DEF-GROUND-12)
- A single string passed as `sources` raises `TypeError` instead of grounding against each character (DEF-GROUND-13)
- The config YAML is parsed once instead of three times per claim, and the lexical corpus and BM25 indexes are built once per content instead of per claim (DEF-PERF-7, DEF-PERF-8)

## [1.0.35] - 2026-06-22

### Changed

- Package description (README) condensed, with an average-performance table; no library code changed

## [1.0.34] - 2026-06-22

### Added

- `groundrails.init()` and `groundrails init`: provision calibration and models from, in order, a per-resource override, S3, a local folder or Hugging Face, and write the resolved runtime config to `groundrails.json`
- `export_calibration()` and `groundrails calibration export`

### Changed

- `ground`, `ground_batch` and `grounding_document` raise `NotInitializedError` until `init()` has run; CLI `ground` reads `groundrails.json` or exits 2 with a hint
- Runtime settings are held in the process instead of in `.stellars-plugins/settings.json`

## [1.0.33] - 2026-06-19

### Changed

- Version-only release; the package is identical to 1.0.32

## [1.0.32] - 2026-06-19

### Added

- The cross-lingual bridge installs a missing argos translation model into English on first need; off when `HF_HUB_OFFLINE` is set or `GROUNDRAILS_ARGOS_AUTO_INSTALL=0`

## [1.0.31] - 2026-06-19

### Added

- `ignore_language` on `ground` and `ground_batch`, and CLI `--ignore-language`, bypass the unsupported-language block
- `groundrails.__version__`

### Changed

- The unsupported-language block applies only when the claim's language differs from the evidence's; a same-language non-English pair grounds directly

### Fixed

- Concurrent WordNet lookups in `ground_batch` no longer crash documents; lookups are serialised, and a failed lookup gives no antonyms instead of an error
- One failing claim in `ground_batch` gives an ungrounded match instead of aborting the batch; `UnsupportedLanguageError` still raises

## [1.0.30] - 2026-06-19

### Added

- Semantic cascade (OpenVINO int8 bge-m3, bge-reranker and mDeBERTa-NLI) as a switch that works with every effort tier, set by `calibration.mode: semantic` or `--semantic 1`; uncertain and cross-lingual claims escalate to it, and a joint head combines the verdicts; extras `[semantic-grounder]` and `[all]`
- `groundrails download` fetches the cascade models ahead of the first run
- `grounding_document()` and `build_grounding_document()`, plus `GroundingMatch.support` and `.grounded`: per claim a verdict, one final score and the supporting quote with its source, line and character offset
- Extracted claims carry their character span in the answer document (`char_start`, `char_end`)

### Changed

- CLI `ground` takes a document and evidence files (`ground DOCUMENT EVIDENCE ...`); `--json` prints the grounding document, `--full-output` keeps the per-scorer output

## [1.0.29] - 2026-06-18

### Added

- First PyPI release: deterministic, torch-free claim grounding (`ground`, `ground_batch`) with `low`, `medium` and `high` effort tiers (regex, Levenshtein and BM25 recall), numeric and entity contradiction checks, calibrated verdicts, claim extraction and an intra-document consistency check
- Cross-lingual bridge on the `high` tier through argos translation; `UnsupportedLanguageError` when a claim is confidently non-English and no translation model is installed
- CLI commands `ground`, `extract-claims`, `check-consistency`, `config` and `setup`

[1.1.2]: https://pypi.org/project/groundrails/1.1.2/
[1.1.1]: https://pypi.org/project/groundrails/1.1.1/
[1.0.38]: https://pypi.org/project/groundrails/1.0.38/
[1.0.35]: https://pypi.org/project/groundrails/1.0.35/
[1.0.34]: https://pypi.org/project/groundrails/1.0.34/
[1.0.33]: https://pypi.org/project/groundrails/1.0.33/
[1.0.32]: https://pypi.org/project/groundrails/1.0.32/
[1.0.31]: https://pypi.org/project/groundrails/1.0.31/
[1.0.30]: https://pypi.org/project/groundrails/1.0.30/
[1.0.29]: https://pypi.org/project/groundrails/1.0.29/
