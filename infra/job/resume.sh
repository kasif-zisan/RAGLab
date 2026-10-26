#!/bin/bash
# Resume a bootstrap that died after the runtime was installed.
# ---------------------------------------------------------------------------
# The box already has the venv, torch+CUDA and the GPU. Re-launching would pay
# ~5 minutes of boot and reinstall for nothing. This picks up at the step that
# failed: fetch input, fetch job, run it.
#
# Runs DETACHED (nohup) because ssm send-command polls for a result and would
# time out long before an embedding benchmark finishes. The log is pushed to S3
# on a timer, same as bootstrap, because the original tailer was killed when the
# first script's EXIT trap fired.
# ---------------------------------------------------------------------------
set -uo pipefail
BUCKET=raglab-zisan-apne1
REGION=ap-northeast-1
WORK=/opt/raglab
JOB=${JOB:-bench_embed.py}
export HF_HOME=$WORK/hf
export AWS_DEFAULT_REGION=$REGION
export TOKENIZERS_PARALLELISM=true

IID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -sX PUT \
       http://169.254.169.254/latest/api/token \
       -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')" \
       http://169.254.169.254/latest/meta-data/instance-id)
LOG=$WORK/resume.log
exec > >(tee -a "$LOG") 2>&1

( while sleep 60; do
    aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID-resume.log" --region "$REGION" >/dev/null 2>&1 || true
  done ) &
TAILER=$!
finish() {
  local rc=$?
  kill "${TAILER:-0}" 2>/dev/null || true
  echo "=== resume exit $rc at $(date -Is) ==="
  aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID-resume.log" --region "$REGION" || true
}
trap finish EXIT

echo "=== resume on $IID at $(date -Is) ==="
mkdir -p "$WORK/chunks"
aws s3 sync "s3://$BUCKET/chunks/" "$WORK/chunks/" --region "$REGION" --only-show-errors
echo "--- chunks ---"; ls -la "$WORK/chunks"
aws s3 cp "s3://$BUCKET/job/$JOB" "$WORK/$JOB" --region "$REGION"
echo "=== running $JOB at $(date -Is) ==="
cd "$WORK" && "$WORK/venv/bin/python" "$WORK/$JOB"
echo "=== $JOB finished at $(date -Is) ==="
