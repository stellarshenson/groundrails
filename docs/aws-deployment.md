# AWS deployment

The lexical grounder runs as a container-image AWS Lambda that provisions itself from S3 at cold start and grounds the claims in each invoke event. The deployment doubles as the project's end-to-end functional test - one script stands the stack up, invokes it, checks the result, and tears it all down. All resources live under [`aws/`](../aws/).

## What runs

- **Handler** - [`aws/handler.py`](../aws/handler.py); cold start calls `groundrails.init(source=$GROUNDRAILS_SOURCE, models="none", ...)`, then each invoke calls `grounding_document`
- **Image** - [`aws/Dockerfile`](../aws/Dockerfile) on `public.ecr.aws/lambda/python:3.12`; lexical deps only ([`aws/requirements-lambda.txt`](../aws/requirements-lambda.txt)), the package installed `--no-deps` from a locally-built wheel
- **No model weights** - the low-effort, same-language lexical path needs none; cold start pulls only the calibration JSON from S3 (SaT + CTranslate2 are HIGH-tier cross-lingual, out of scope here)

## Cold-start provisioning

- **`GROUNDRAILS_SOURCE`** - the S3 base (e.g. `s3://groundrails-dev`); `init` pulls `calibration.json` from it
- **`GROUNDRAILS_EFFORT`** - the lexical tier (`low`)
- **Readiness** - `init` marks the grounder ready in-process; `grounding_document` raises `NotInitializedError` if it never ran
- **Role permissions** - `s3:GetObject` + `s3:ListBucket` on the source bucket, plus `AWSLambdaBasicExecutionRole`

## Lambda Design

A working handler initializes once at cold start (the container stays warm across invokes), then grounds each event. `init` runs in-process - it marks the grounder ready and provisions the calibration JSON from S3; `grounding_document` raises `NotInitializedError` if it never ran.

```python
import os

import groundrails
from groundrails.config import load_document_processing_config

_CFG = None  # built once, reused across warm invokes


def _ready(effort: str) -> None:
    global _CFG
    if _CFG is not None:
        return
    src = os.environ.get("GROUNDRAILS_SOURCE")  # e.g. s3://my-bucket
    if src:
        groundrails.init(
            source=src,            # init pulls <src>/calibration.json
            models="none",         # lexical path needs no model weights
            wordnet=False,
            home="/tmp/groundrails",  # the only writable path in Lambda
            aws_region=os.environ.get("AWS_REGION"),
        )
    _CFG = load_document_processing_config().overlay(lexical_effort=effort)


def handler(event, context):
    effort = (event or {}).get("effort") or os.environ.get("GROUNDRAILS_EFFORT", "low")
    _ready(effort)
    claims = (event or {}).get("claims") or []
    raw = (event or {}).get("sources") or []
    sources = [tuple(s) if isinstance(s, (list, tuple)) else s for s in raw]
    doc = groundrails.grounding_document(
        claims, sources, config=_CFG, semantic=False, ignore_language=True, max_workers=1
    )
    return {"ok": True, "effort": effort, "grounding": doc}
```

- **`home="/tmp/groundrails"`** - Lambda's task root is read-only; point `GROUNDRAILS_HOME` (here via the `home` argument) at `/tmp`, the only writable path. `groundrails.json` and the provisioned calibration land there
- **`ignore_language=True`** - the bridge guard is off; this lexical function is same-language only
- **`max_workers=1`** - one Lambda invocation grounds serially; concurrency comes from Lambda scaling, not threads

### Where to put resources in S3

`GROUNDRAILS_SOURCE` is the base; `init` resolves `<source>/calibration.json` and, when `--models` points at S3, `<models>/<name>/`:

```text
s3://my-bucket/                       # GROUNDRAILS_SOURCE = s3://my-bucket
  calibration.json                    # required - the provisioned calibration block
  models/                             # optional - only for the semantic cascade / cross-lingual
    bge-m3/openvino_model.{xml,bin}
    bge-reranker/openvino_model.{xml,bin}
    mdeberta-nli/openvino_model.{xml,bin}
    sat/openvino_model.{xml,bin}      # the SaT segmenter, for the cross-lingual MT bridge
```

- **Lexical only** - stage just `calibration.json`; leave `models="none"` and the bucket needs nothing else
- **Semantic / cross-lingual** - also stage the int8 IRs under `models/` and pass `--models s3://my-bucket/models` to `init`; size the function with a larger image or EFS rather than S3-to-`/tmp`
- **Role** - the execution role needs `s3:GetObject` + `s3:ListBucket` on the bucket

## The functional test (local only)

[`aws/e2e.sh`](../aws/e2e.sh) runs the full cycle and is the project's end-to-end acceptance gate. It needs AWS credentials, so it runs locally only - never in CI.

```bash
aws/e2e.sh all          # build -> push -> stage -> deploy -> invoke -> teardown
KEEP=1 aws/e2e.sh all   # keep the stack up after the invoke
aws/e2e.sh {build|push|deploy|invoke|teardown}   # sub-steps
```

Resources it creates (and tears down): an ECR repo, an S3 bucket with `calibration.json` staged, an IAM execution role, and the Lambda function. Defaults target the **kolomolo** account in **eu-central-1** with the `groundrails-dev` name for the bucket / repo / function; override via env (`AWS_PROFILE`, `AWS_REGION`, `BUCKET`, `REPO`, `FN`, `ROLE`, `TAG`).

## Verified result

A run grounds two claims against one source:

- "The Eiffel Tower is in Paris." → grounded (fuzzy 0.732), support located
- "The tower is 2000 metres tall." → contradicted, numeric conflict `2000` vs `330`

## Caveats

- **`teardown` deletes the bucket and everything in it** - point it at a throwaway `BUCKET=` for the test
- **Lexical only** - the semantic cascade (~1.4 GB int8 IRs) and the HIGH-tier cross-lingual MT bridge are out of scope; for those, mirror the IRs / SaT IR to S3 and pass `--models s3://…/models` to `init`
- **Working-tree wheel** - the image carries the locally-built groundrails (`uv build --wheel`), not the PyPI release
