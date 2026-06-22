# CLI reference

The `groundrails` command-line surface - grounding claims, extracting them, and the support commands. The Python functions live in [`api-reference.md`](api-reference.md). The core needs no extras; `--semantic 1` needs `groundrails[semantic-grounder]`.

## Commands

| Command | What it does |
|---|---|
| `groundrails init [OPTIONS]` | provision calibration + models (S3 / local / HF) and write `groundrails.json`; run once before grounding |
| `groundrails ground DOCUMENT EVIDENCE...` | extract claims from the one document, ground them against the evidence |
| `groundrails ground --claims FILE EVIDENCE...` | ground a structured claims file |
| `groundrails ground --claim TEXT [--claim TEXT] EVIDENCE...` | ground inline claim(s), repeatable |
| `groundrails extract-claims --document DOC` | pull claims (with their locations) from a document |
| `groundrails check-consistency --document DOC` | intra-document contradiction report |
| `groundrails calibration export -o FILE` | write the active calibration block to a provisioned JSON |
| `groundrails config` / `download` / `setup` | print config / fetch the cascade models / show built-in settings |

- **Initialize first** - grounding is gated: run `groundrails init` once (it writes `groundrails.json`); a `ground` with no `groundrails.json` exits 2 with an init hint
- **`$GROUNDRAILS_HOME`** - where `init` writes `groundrails.json` and the provisioned assets; unset → `groundrails.json` falls back to `./` (the current directory) and assets to `~/.cache/groundrails`. In Lambda, set it to a writable path (`/tmp/groundrails`)
- **Flags** - `--json` (grounding document), `--full-output` (per-scorer detail), `--semantic 1` (add the cascade), `--effort {low,medium,high}`
- **Exit code** - 0 if every claim is grounded, 1 if any is not (usable as a CI gate)

## Examples

Provision once - the resource chain resolves per override, then S3, then a local folder, then HuggingFace; `init` writes `groundrails.json`:

```bash
groundrails init                                                  # built-in defaults (bundled calibration, lazy HF)
groundrails init --source s3://groundrails-data/prod              # pull calibration + models from S3
groundrails init --calibration ./calibration.json --models ./models   # from a local folder
groundrails init --source s3://bucket/base --aws-profile my-profile \
  --aws-endpoint-url https://s3.example.com:9000                       # S3-compatible endpoint (e.g. MinIO / RustFS)
```

Export the active calibration to the provisioned JSON an `init` reads:

```bash
groundrails calibration export -o calibration.json
```

Default: extract claims from the answer, check against evidence, emit the grounding document:

```bash
groundrails ground answer.md evidence.txt --json
```

Inline claims, repeatable; the remaining positionals are always evidence:

```bash
groundrails ground --claim "The tower is in Paris." --claim "It is 330 m tall." evidence.txt
```

A structured claims file against several evidence sources:

```bash
groundrails ground --claims claims.json evidence1.txt evidence2.txt
```

Pull claims out of a document first, then ground that file:

```bash
groundrails extract-claims --document answer.md > claims.json
groundrails ground --claims claims.json evidence.txt --json
```

Deeper check and cross-lingual claims - install the extra, fetch the models once, then add `--semantic 1`:

```bash
pip install 'groundrails[semantic-grounder]'
groundrails download
groundrails ground answer.md evidence.txt --semantic 1 --effort high
```

Gate a pipeline on the exit code - non-zero means at least one claim was not grounded:

```bash
if groundrails ground answer.md evidence.txt --json > grounding.json; then
  echo "all claims grounded"
else
  echo "ungrounded claims - blocking" >&2
  exit 1
fi
```
