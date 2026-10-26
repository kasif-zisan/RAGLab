#!/bin/bash
# Embed both selected arms, then power off. Separate stores, one after the other.
#   * SEPARATE vector stores per arm: a single store holding both would make a
#     retrieval hit unattributable to a parser, which is the whole experiment.
#   * SEQUENTIAL, not parallel: one T4, and running them together would halve
#     each one's throughput while making the progress numbers meaningless.
#   * Shuts down at the end. The deadman re-armed below fires regardless, in
#     case this script never reaches its last line.
set -uo pipefail
BUCKET=raglab-zisan-apne1; REGION=ap-northeast-1; WORK=/opt/raglab
export HF_HOME=$WORK/hf AWS_DEFAULT_REGION=$REGION TOKENIZERS_PARALLELISM=true
export MODEL="${MODEL:-Qwen/Qwen3-Embedding-0.6B}"

IID=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -sX PUT \
       http://169.254.169.254/latest/api/token \
       -H 'X-aws-ec2-metadata-token-ttl-seconds: 300')" \
       http://169.254.169.254/latest/meta-data/instance-id)
LOG=$WORK/embed.log
exec > >(tee -a "$LOG") 2>&1
( while sleep 60; do
    aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID-embed.log" --region "$REGION" >/dev/null 2>&1 || true
  done ) &
TAILER=$!
finish() {
  local rc=$?
  kill "${TAILER:-0}" 2>/dev/null || true
  echo "=== run_both exit $rc at $(date -Is) ==="
  aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID-embed.log" --region "$REGION" || true
  shutdown -h now
}
trap finish EXIT

echo "=== embedding both arms with $MODEL at $(date -Is) ==="
for ARM in B1_3200_ov889 A_2400_ov360; do
  echo; echo "########## $ARM ##########"
  ARM=$ARM "$WORK/venv/bin/python" "$WORK/embed.py" || echo "!!! $ARM exited non-zero"
done
echo "=== both arms finished at $(date -Is) ==="
