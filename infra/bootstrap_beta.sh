#!/bin/bash
# EC2 user-data, Phase 0 variant: identical to bootstrap.sh except ONLY_DOCS
# limits the corpus sync to one document. Runs as root on first boot.
#
# The saving is iteration time, not money - in-region S3 to EC2 transfer is
# free and takes seconds. The cost is that this no longer exercises the same
# sync path Phase 1 will use.
#
# Prepares the machine, then fetches and runs s3://$BUCKET/job/run.py.
# The job is NOT baked in here - Phase 0 (validate on 2 pages) and Phase 1
# (parse 196 documents) use this same bootstrap with a different run.py.

set -euo pipefail

BUCKET=raglab-zisan-apne1
REGION=ap-northeast-1
WORK=/opt/raglab
SHUTDOWN_WHEN_DONE=false         # leave the box up afterwards so you can SSM in and look around
MAX_LIFETIME_MIN=240             # deadman switch: the box dies after this no matter what
ONLY_DOCS="ORLY_2015"            # Phase 0: one document. Empty string = whole corpus.

export HF_HOME=$WORK/hf
DOCLING_CACHE=/root/.cache/docling
export AWS_DEFAULT_REGION=$REGION

mkdir -p "$WORK" "$HF_HOME" "$DOCLING_CACHE"
LOG=$WORK/bootstrap.log
exec > >(tee -a "$LOG") 2>&1

TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token \
          -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
IID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
        http://169.254.169.254/latest/meta-data/instance-id)

MODELS_KEY="s3://$BUCKET/models/docling-models.tar"

# Runs on EVERY exit path - success, job failure, or a typo on line 3. Seeds the
# model cache first (so a failed parse still leaves the weights for next time),
# then ships the log, then powers off. A dead instance cannot tell you why it died.
finish() {
  local rc=$?
  echo "=== exit $rc at $(date -Is) ==="
  if ! aws s3 ls "$MODELS_KEY" --region "$REGION" >/dev/null 2>&1; then
    tar -cf /tmp/m.tar -C / "${HF_HOME#/}" "${DOCLING_CACHE#/}" 2>/dev/null || true
    SIZE=$(stat -c%s /tmp/m.tar 2>/dev/null || echo 0)
    # Only seed if we actually got weights. An empty tar uploaded once would make
    # every future instance restore nothing and never notice, because restoring
    # nothing is not an error.
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
    echo "===   terminate now:  sudo shutdown -h now   (or from your laptop: aws ec2 terminate-instances) ==="
  fi
}
trap finish EXIT

# With SHUTDOWN_WHEN_DONE=false nothing powers the box off at the end, so a
# forgotten instance bills until you notice. Schedule the kill up front instead:
# it fires whether the job hung, finished, or you walked away. Cancel it from
# inside the box with:  sudo shutdown -c
shutdown -h +$MAX_LIFETIME_MIN "raglab deadman: $MAX_LIFETIME_MIN min from boot" || true

echo "=== $IID starting at $(date -Is) ==="
echo "=== deadman: this instance terminates at $(date -Is -d "+$MAX_LIFETIME_MIN minutes") ==="

# ---- 1. runtime ----------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3.12-venv unzip curl
# docling's table model imports OpenCV, which links against system graphics
# libraries that a headless server image does not carry. Without these:
# ImportError: libGL.so.1 - at pipeline init, before a single page is read.
apt-get install -y -qq libgl1 libglib2.0-0t64 || apt-get install -y -qq libgl1 libglib2.0-0

curl -s https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp && /tmp/aws/install --update
aws --version

python3 -m venv "$WORK/venv"
# CPU-only torch on purpose: the default wheel drags in ~2.5 GB of CUDA we have
# no GPU to use. Installing it first means docling finds torch already satisfied.
"$WORK/venv/bin/pip" install -q --upgrade pip
"$WORK/venv/bin/pip" install -q --index-url https://download.pytorch.org/whl/cpu torch torchvision
"$WORK/venv/bin/pip" install -q docling boto3 pymupdf

# ---- 2. model weights (restore only; seeding happens in the trap) ---------
if aws s3 ls "$MODELS_KEY" --region "$REGION" >/dev/null 2>&1; then
  echo "--- models: restoring from S3 ---"
  aws s3 cp "$MODELS_KEY" /tmp/m.tar --region "$REGION"
  tar -xf /tmp/m.tar -C /
else
  echo "--- models: not in S3; docling will download them during convert() ---"
fi

# ---- 3. corpus -----------------------------------------------------------
mkdir -p "$WORK/pdf"
aws s3 cp "s3://$BUCKET/manifest.json" "$WORK/manifest.json" --region "$REGION"
if [ -z "$ONLY_DOCS" ]; then
  aws s3 sync "s3://$BUCKET/pdf/" "$WORK/pdf/" --region "$REGION" --only-show-errors
else
  for stem in $ONLY_DOCS; do
    aws s3 cp "s3://$BUCKET/pdf/$stem.pdf" "$WORK/pdf/$stem.pdf" --region "$REGION"
  done
fi
echo "--- corpus: $(ls "$WORK/pdf" | wc -l) pdfs ---"

# ---- 4. the job ----------------------------------------------------------
aws s3 cp "s3://$BUCKET/job/run.py" "$WORK/run.py" --region "$REGION"
echo "=== running job at $(date -Is) ==="
cd "$WORK" && "$WORK/venv/bin/python" "$WORK/run.py"
echo "=== job finished at $(date -Is) ==="
