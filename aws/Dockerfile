# Container image for the lexical groundrails Lambda.
# Build context is the repo root so the locally-built wheel under dist/ is in scope:
#   uv build --wheel
#   docker build -f aws/Dockerfile -t groundrails-dev .
FROM public.ecr.aws/lambda/python:3.12

# Lean runtime deps (the package is installed --no-deps; lexical path only).
COPY aws/requirements-lambda.txt ./
RUN pip install --no-cache-dir -r requirements-lambda.txt

# The locally-built groundrails wheel (carries the bundled calibration YAML).
COPY dist/*.whl ./
RUN pip install --no-cache-dir --no-deps ./*.whl && rm -f ./*.whl

COPY aws/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
