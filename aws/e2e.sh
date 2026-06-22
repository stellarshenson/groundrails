#!/usr/bin/env bash
# End-to-end AWS functional test for the lexical groundrails Lambda:
#   build -> push to ECR -> stage calibration on S3 -> deploy Lambda ->
#   invoke -> collect grounding results -> tear everything down.
#
# Local only - this is the project's end-to-end acceptance gate (see
# docs/acc-criteria-grounder.md). It needs AWS credentials, so it never runs
# in CI; run it by hand against a throwaway account/bucket.
#
# Usage:
#   aws/e2e.sh all        # full cycle + teardown (the functional test)
#   aws/e2e.sh deploy     # build, push, bucket, stage, role, lambda
#   aws/e2e.sh invoke     # invoke the deployed function with event.example.json
#   aws/e2e.sh teardown   # delete lambda, role, ecr repo, bucket
#
# Config via env (defaults target the kolomolo account):
set -euo pipefail

PROFILE="${AWS_PROFILE:-kolomolo}"
REGION="${AWS_REGION:-eu-central-1}"
BUCKET="${BUCKET:-groundrails-dev}"
REPO="${REPO:-groundrails-dev}"
FN="${FN:-groundrails-dev}"
ROLE="${ROLE:-groundrails-dev-lambda}"
TAG="${TAG:-latest}"

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
AWS="aws --profile $PROFILE --region $REGION"
ACCOUNT="$($AWS sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

log() { echo -e "\033[1;36m[e2e]\033[0m $*" >&2; }

build() {
  log "building wheel + image ($IMAGE)"
  ( cd "$ROOT" && rm -f dist/*.whl && uv build --wheel >/dev/null )
  ( cd "$ROOT" && DOCKER_BUILDKIT=0 docker build -f aws/Dockerfile -t "$REPO:$TAG" . >/dev/null )
}

push() {
  log "ensuring ECR repo + pushing image"
  $AWS ecr describe-repositories --repository-names "$REPO" >/dev/null 2>&1 \
    || $AWS ecr create-repository --repository-name "$REPO" >/dev/null
  $AWS ecr get-login-password | docker login --username AWS --password-stdin "$REGISTRY" >/dev/null 2>&1
  docker tag "$REPO:$TAG" "$IMAGE"
  docker push "$IMAGE" >/dev/null
  log "pushed $IMAGE"
}

bucket() {
  log "ensuring S3 bucket s3://$BUCKET"
  $AWS s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1 || \
    $AWS s3api create-bucket --bucket "$BUCKET" \
      --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
}

stage() {
  log "exporting + staging calibration.json to s3://$BUCKET/calibration.json"
  local tmp; tmp="$(mktemp -d)/calibration.json"
  ( cd "$ROOT" && uv run groundrails calibration export -o "$tmp" )
  $AWS s3 cp "$tmp" "s3://$BUCKET/calibration.json" >/dev/null
}

role() {
  log "ensuring IAM execution role $ROLE"
  if ! $AWS iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    $AWS iam create-role --role-name "$ROLE" \
      --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
    $AWS iam attach-role-policy --role-name "$ROLE" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null
    $AWS iam put-role-policy --role-name "$ROLE" --policy-name s3-read \
      --policy-document "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:ListBucket\"],\"Resource\":[\"arn:aws:s3:::$BUCKET\",\"arn:aws:s3:::$BUCKET/*\"]}]}" >/dev/null
    log "waiting 12s for role propagation"; sleep 12
  fi
}

deploy_fn() {
  local role_arn; role_arn="$($AWS iam get-role --role-name "$ROLE" --query Role.Arn --output text)"
  if $AWS lambda get-function --function-name "$FN" >/dev/null 2>&1; then
    log "updating function code"
    $AWS lambda update-function-code --function-name "$FN" --image-uri "$IMAGE" >/dev/null
  else
    log "creating function $FN (retry until role assumable)"
    for i in $(seq 1 6); do
      if $AWS lambda create-function --function-name "$FN" \
          --package-type Image --code "ImageUri=$IMAGE" --role "$role_arn" \
          --architectures x86_64 --timeout 60 --memory-size 1024 \
          --environment "Variables={GROUNDRAILS_SOURCE=s3://$BUCKET,GROUNDRAILS_EFFORT=low}" >/dev/null 2>&1; then
        break
      fi
      log "  role not ready, retry $i/6"; sleep 8
    done
  fi
  log "waiting for function active"
  $AWS lambda wait function-active-v2 --function-name "$FN"
}

deploy() { build; push; bucket; stage; role; deploy_fn; log "deploy complete"; }

invoke() {
  log "invoking $FN with aws/event.example.json"
  local out; out="$(mktemp)"
  $AWS lambda invoke --function-name "$FN" \
    --payload "fileb://$HERE/event.example.json" \
    --cli-binary-format raw-in-base64-out "$out" >/dev/null
  echo "----- grounding result -----"
  python3 -m json.tool < "$out"
  echo "----------------------------"
}

teardown() {
  log "tearing down"
  $AWS lambda delete-function --function-name "$FN" >/dev/null 2>&1 && log "  deleted lambda" || true
  if $AWS iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
    $AWS iam delete-role-policy --role-name "$ROLE" --policy-name s3-read >/dev/null 2>&1 || true
    $AWS iam detach-role-policy --role-name "$ROLE" \
      --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole >/dev/null 2>&1 || true
    $AWS iam delete-role --role-name "$ROLE" >/dev/null 2>&1 && log "  deleted role" || true
  fi
  $AWS ecr delete-repository --repository-name "$REPO" --force >/dev/null 2>&1 && log "  deleted ecr repo" || true
  $AWS s3 rm "s3://$BUCKET" --recursive >/dev/null 2>&1 || true
  $AWS s3api delete-bucket --bucket "$BUCKET" >/dev/null 2>&1 && log "  deleted bucket" || true
}

case "${1:-all}" in
  build) build ;;
  push) push ;;
  deploy) deploy ;;
  invoke) invoke ;;
  teardown) teardown ;;
  all) deploy; invoke; [ "${KEEP:-0}" = "1" ] || teardown; log "e2e done" ;;
  *) echo "usage: $0 {build|push|deploy|invoke|teardown|all}" >&2; exit 2 ;;
esac
