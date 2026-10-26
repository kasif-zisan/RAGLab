#!/bin/bash
# EC2 user-data for the Phase 7 embedding worker (GPU). Runs as root on boot.
# ---------------------------------------------------------------------------
# Derived from bootstrap.sh. Same skeleton - prepare box, fetch job from S3, run
# it, ship the log, self-terminate. See infra/README.md.
#
# WHAT DIFFERS FROM bootstrap.sh, AND WHY:
#   * torch comes from the DEFAULT index, not /whl/cpu. bootstrap.sh forces the
#     CPU wheel because that box had no GPU and the CUDA wheel is ~2.5 GB of dead
#     weight. Here the CUDA runtime is the entire point - installing the CPU
#     wheel on a g4dn would run the job on 4 vCPUs and look merely "slow", with
#     no error anywhere.
#   * NO OMP/OPENBLAS thread pinning. On the parse box those limits fixed load
#     average 124 from N worker processes each spawning a pool. Here there is ONE
#     process and the GPU does the arithmetic; pinning to 1 thread would only
#     starve tokenisation, which is the CPU-side bottleneck.
#   * No libgl1 - nothing here imports cv2.
#   * SHUTDOWN_WHEN_DONE=false: this first boot installs and benchmarks, and the
#     box is meant to be inspected afterwards. The deadman timer is still armed,
#     so forgetting about it costs at most MAX_LIFETIME_MIN of spot time.
# ---------------------------------------------------------------------------

set -euo pipefail

BUCKET=raglab-zisan-apne1
REGION=ap-northeast-1
WORK=/opt/raglab
JOB=${JOB:-bench_embed.py}
SHUTDOWN_WHEN_DONE=${SHUTDOWN:-false}   # false = leave up to inspect; true = power off when done
MAX_LIFETIME_MIN=${LIFETIME:-120}       # deadman, minutes from boot

export HF_HOME=$WORK/hf
export AWS_DEFAULT_REGION=$REGION
export TOKENIZERS_PARALLELISM=true    # tokenising is the CPU-side bottleneck here

mkdir -p "$WORK" "$HF_HOME"
LOG=$WORK/bootstrap.log
exec > >(tee -a "$LOG") 2>&1

TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token \
          -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-id)

MODELS_KEY="s3://$BUCKET/models/hf-embed.tar"

# Runs on EVERY exit path. A dead instance cannot tell you why it died.
finish() {
  local rc=$?
  kill "${TAILER:-0}" 2>/dev/null || true
  echo "=== exit $rc at $(date -Is) ==="
  if ! aws s3 ls "$MODELS_KEY" --region "$REGION" >/dev/null 2>&1; then
    tar -cf /tmp/m.tar -C / "${HF_HOME#/}" 2>/dev/null || true
    SIZE=$(stat -c%s /tmp/m.tar 2>/dev/null || echo 0)
    # Only seed above a threshold. An empty tar uploaded once would make every
    # future instance "restore" nothing and never notice - restoring nothing is
    # not an error.
    if [ "$SIZE" -gt 100000000 ]; then
      aws s3 cp /tmp/m.tar "$MODELS_KEY" --region "$REGION" && echo "seeded models: $SIZE bytes"
    else
      echo "model cache only $SIZE bytes - NOT seeding S3"
    fi
  fi
  aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID.log" --region "$REGION" || true
  if $SHUTDOWN_WHEN_DONE; then
    shutdown -h now
  else
    echo "=== instance left RUNNING for inspection. deadman still armed. ==="
    echo "===   infra/ssm.sh run $IID '...'   /   infra/ssm.sh kill $IID ==="
  fi
}
trap finish EXIT

( while sleep 120; do
    aws s3 cp "$LOG" "s3://$BUCKET/logs/$IID.log" --region "$REGION" >/dev/null 2>&1 || true
  done ) &
TAILER=$!

# Fires regardless of what the job does. The end-of-script shutdown only runs if
# the script REACHES the end; a hung job never does. Cancel with: sudo shutdown -c
shutdown -h +$MAX_LIFETIME_MIN "raglab deadman: $MAX_LIFETIME_MIN min from boot" || true

echo "=== $IID starting at $(date -Is) ==="
echo "=== deadman: terminates at $(date -Is -d "+$MAX_LIFETIME_MIN minutes") ==="

# ---- 0. prove the GPU is actually there ----------------------------------
# A g4dn whose driver did not come up still boots, still runs, and torch simply
# reports cuda unavailable - the job then silently runs on CPU at ~1/20 the
# speed. Fail loudly here instead.
echo "--- nvidia-smi ---"
nvidia-smi || { echo "FATAL: no NVIDIA driver on a GPU instance"; exit 1; }

# ---- 1. runtime ----------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3-venv unzip curl

curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update
aws --version

python3 -m venv "$WORK/venv"
PIP="$WORK/venv/bin/pip"
$PIP install -q --upgrade pip
# Default index on purpose - this wheel carries the CUDA runtime.
$PIP install -q torch
$PIP install -q sentence-transformers transformers einops boto3 numpy
echo "--- installed ---"
"$WORK/venv/bin/python" - <<'PY'
import torch, sentence_transformers, transformers
print("torch               ", torch.__version__)
print("cuda available      ", torch.cuda.is_available())
print("device              ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("sentence-transformers", sentence_transformers.__version__)
print("transformers        ", transformers.__version__)
assert torch.cuda.is_available(), "torch cannot see the GPU"
PY

# ---- 2. model weights (restore only; seeding happens in the trap) ---------
if aws s3 ls "$MODELS_KEY" --region "$REGION" >/dev/null 2>&1; then
  echo "--- models: restoring from S3 ---"
  aws s3 cp "$MODELS_KEY" /tmp/m.tar --region "$REGION"
  tar -xf /tmp/m.tar -C /
else
  echo "--- models: not cached; will download from HuggingFace ---"
fi

# ---- 3. input ------------------------------------------------------------
mkdir -p "$WORK/chunks"
aws s3 sync "s3://$BUCKET/chunks/" "$WORK/chunks/" --region "$REGION" --only-show-errors
echo "--- chunks: $(ls -la "$WORK/chunks" | grep -c jsonl.gz || true) file(s) ---"
ls -la "$WORK/chunks"

# ---- 4. the job ----------------------------------------------------------
aws s3 cp "s3://$BUCKET/job/$JOB" "$WORK/$JOB" --region "$REGION"
echo "=== running $JOB at $(date -Is) ==="
cd "$WORK" && "$WORK/venv/bin/python" "$WORK/$JOB"
echo "=== job finished at $(date -Is) ==="
