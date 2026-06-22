# groundrails on AWS Lambda

Deploys the lexical groundrails grounder as a container-image Lambda and runs it
end to end. It runs the low-effort, same-language lexical path: claims arrive
already split, and recall is exact / fuzzy / BM25 fused by a frozen logistic -
plain coefficients carried in the calibration JSON, not a neural model. So the
only thing cold start pulls from S3 via `groundrails.init` is that calibration
JSON; no model weights. The SaT sentence segmenter and the CTranslate2
translator belong to the HIGH-tier cross-lingual MT bridge, which is out of
scope here. Each invoke grounds the event's claims against its sources and
returns the grounding document.

## Layout

- `handler.py` - the Lambda handler; cold-start `init` from S3, then `grounding_document`
- `Dockerfile` - container image on `public.ecr.aws/lambda/python:3.12`, lexical deps only
- `requirements-lambda.txt` - the lean runtime deps (package installed `--no-deps`)
- `event.example.json` - a sample invoke payload
- `e2e.sh` - build → push → stage → deploy → invoke → collect → teardown

## End-to-end functional test

This is the project's end-to-end acceptance gate (see
[`../docs/acc-criteria-grounder.md`](../docs/acc-criteria-grounder.md)). It
needs AWS credentials, so it runs **locally only - never in CI**. One command
builds the image, pushes it to ECR, stages the calibration JSON on S3, deploys
the Lambda, invokes it, prints the grounding result, and deletes every resource
it created:

```bash
aws/e2e.sh all          # full cycle + teardown (the functional test)
KEEP=1 aws/e2e.sh all   # keep the stack up after the invoke
```

Sub-steps are available too: `aws/e2e.sh {build|push|deploy|invoke|teardown}`.

Defaults target the **kolomolo** account in **eu-central-1** with the
`groundrails-dev` bucket / ECR repo / function / role; override via env
(`AWS_PROFILE`, `AWS_REGION`, `BUCKET`, `REPO`, `FN`, `ROLE`, `TAG`).

The cold-start `init` is wired by two function env vars: `GROUNDRAILS_SOURCE`
(the S3 base, e.g. `s3://groundrails-dev`) and `GROUNDRAILS_EFFORT` (`low`).

## Verified result

A run grounds two claims against one source:

- "The Eiffel Tower is in Paris." → grounded (fuzzy 0.732), support located in the evidence
- "The tower is 2000 metres tall." → contradicted, numeric conflict `2000` vs `330`

## Caveats

- **`e2e.sh teardown` deletes the `groundrails-dev` bucket and everything in it.**
  Do not point it at a bucket holding data you want to keep; use a throwaway
  `BUCKET=` for the test if your project syncs real data there.
- Lexical only - the semantic cascade (~1.4 GB int8 IRs) is out of scope here;
  for that, mirror the IRs to S3 and pass `--models s3://…/models` to `init`,
  and size the Lambda with EFS or a larger image rather than S3-to-/tmp.
- Same-language, low effort - the function runs `effort=low` with
  `ignore_language=True`, so the HIGH-tier cross-lingual MT bridge (the SaT
  segmenter + CTranslate2 translator) never loads; a cross-lingual deployment
  would mirror the SaT IR + argos models and run `effort=high`.
- The image is built with the locally-built wheel (`uv build --wheel`), so it
  carries the working-tree groundrails, not the PyPI release.
