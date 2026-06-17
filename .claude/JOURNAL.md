# Claude Code Journal

This journal tracks substantive work on documents, diagrams, and documentation content.

---

1. **Task - Migrate grounding assets into groundrails** (v0.1.0): relocated all grounding content from the parent `claude-code-plugins` repo into this submodule and aligned `pyproject.toml` with the grounder stack<br>
    **Result**: moved `experiments/grounding/` (round scripts, harness, `synth_mt.py`), 5 notebooks, 2 scripts, `references/`, the data parquet, the `sat-3l-sm-ov` OpenVINO model, and the grounding subset of `docs/` - marketplace docs (`testing_claude_cassettes.md`, `autobuild_lessons_learned.md`, article_01) stay in the parent. Sibling layout preserved so relative paths still resolve. Extended `.gitignore` to shield `/models/*` and `experiments/grounding/{private-rag-forensics,cache,logs}`. Rebuilt `pyproject.toml` `dependencies` with the grounder runtime stack (data/ML, lexical tiers, MT bridge, calibration) and added experiment tooling to `dev`. Dropped the parent's sibling-plugin deps and any `stellars-claude-code-plugins` dependency - the lexical module migrates into `src/groundrails` next, when the `stellars_claude_code_plugins` imports resolve.
