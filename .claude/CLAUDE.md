<!-- Import workspace-level CLAUDE.md configuration -->
<!-- See /home/lab/workspace/.claude/CLAUDE.md for complete rules -->

# Project-Specific Configuration

This file extends workspace-level configuration with project-specific rules. See `/home/lab/workspace/.claude/CLAUDE.md` for the inherited base.

## Project Context

`groundrails` - grounding guardrails for agentic RAG: deterministic, torch-free claim verification. Spun out of the lexical-grounding experiments in the parent `claude-code-plugins` repo (see that repo's `docs/experiments/lexical-grounding-experiments.md` and `docs/experiments/lexical-grounding-sota.md` for the research history through Round 12).

- **Layout** - copier-data-science scaffold (`module_name: groundrails`, `src/`, `tests/`, `notebooks/`, `data/`, `models/`, `reports/`)
- **Environment** - `uv`, Python 3.12; use the Makefile targets (`make install`, `make test`) not bare `pip`/`uv` commands
- **Lint/format** - `ruff`
- **Testing** - `pytest`
- **Data storage** - S3, `stellars-tech` AWS profile, bucket `groundrails-data`; env encryption on
- **Git** - submodule of `claude-code-plugins`, remote `https://github.com/stellarshenson/groundrails.git`
- **License** - MIT
