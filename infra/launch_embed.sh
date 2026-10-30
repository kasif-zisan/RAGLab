#!/bin/bash
# Launch the Phase 7 embedding box (GPU spot). Prints the instance id.
# ---------------------------------------------------------------------------
# Derived from launch.sh. See infra/README.md.
#
# WHAT DIFFERS FROM launch.sh, AND WHY:
#   * AMI is resolved from an SSM PUBLIC PARAMETER, not pinned. launch.sh pins
#     ami-0739e5f3b35c608e5 and its own header says not to - a pinned id ages out
#     of patch coverage and is region-specific.
#   * Deep Learning Base OSS Nvidia Driver AMI (Ubuntu 24.04). Base = driver and
#     CUDA only, no frameworks; torch comes from pip. Ubuntu 24.04 deliberately
#     matches the parse box so apt package names stay valid.
#   * Root volume 100 GB. The AMI's own snapshot is 75 GB - asking for the 40 GB
#     that launch.sh uses makes run-instances fail outright.
#   * g4dn.xlarge: 4 vCPU against the G/VT SPOT quota of 8 (approved 2026-10-08).
#     The On-Demand G/VT quota is 0, which is a SEPARATE quota - a zero in one
#     says nothing about the other.
# ---------------------------------------------------------------------------
#
#   infra/launch_embed.sh                     # g4dn.xlarge spot, AZ 1d
#   infra/launch_embed.sh --az 1a             # different capacity pool after a reclaim
#   infra/launch_embed.sh --job embed.py      # run the real job instead of the benchmark
#
# Safe to re-run: the worker skips shards that already exist in S3.

set -euo pipefail
cd "$(dirname "$0")/.."
echo "raglab phase 7 embedding launcher"          # before anything that can fail
P=(--profile zisan --region ap-northeast-1)

AZ=1d; TYPE=g4dn.xlarge; JOB=bench_embed.py; ARM=""; SHUTDOWN=false; LIFETIME=240
MODEL=${MODEL:-Qwen/Qwen3-Embedding-0.6B}
while [ $# -gt 0 ]; do
  case "$1" in
    --az)   AZ=$2;  shift 2 ;;
    --job)  JOB=$2; shift 2 ;;
    --arm)  ARM=$2; SHUTDOWN=true; shift 2 ;;   # a single-arm run is unattended
    --shutdown) SHUTDOWN=true; shift ;;         # unattended without an --arm
    --lifetime) LIFETIME=$2; shift 2 ;;         # deadman minutes; keep it near the job length
    --keep) SHUTDOWN=false; shift ;;
    *)     TYPE=$1; shift ;;
  esac
done

# A CPU instance has no use for the 75 GB Deep Learning AMI and its CUDA stack.
case "$TYPE" in
  g*|p*|inf*|trn*) AMI_PARAM=/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id; VOL=100; ALLOW_CPU=0 ;;
  *)               AMI_PARAM=/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id; VOL=40; ALLOW_CPU=1 ;;
esac
AMI=$(aws ssm get-parameter "${P[@]}" --name "$AMI_PARAM" --query 'Parameter.Value' --output text)
SG=sg-07468a8f471d43f07
case "$AZ" in
  1a) SUBNET=subnet-0854039f1813e07ab ;;
  1c) SUBNET=subnet-051959a17410f548f ;;
  1d) SUBNET=subnet-09e0868af873fc031 ;;
  *)  echo "unknown AZ: $AZ (use 1a, 1c or 1d)"; exit 1 ;;
esac

# --start-time is REQUIRED here. Without it the API returns only the oldest
# retained page, the AZ filter matches nothing, and the query yields a literal
# "None" that prints as "$None/hr" - a wrong price, not an error.
PRICE=$(aws ec2 describe-spot-price-history "${P[@]}" --instance-types "$TYPE" \
          --product-descriptions "Linux/UNIX" \
          --start-time "$(date -u -d '-1 hour' +%Y-%m-%dT%H:%M:%SZ)" \
          --availability-zone "ap-northeast-$AZ" \
          --query 'SpotPriceHistory[0].SpotPrice' \
          --output text 2>/dev/null | head -1 || echo "?")
# `aws s3 ls` exits 1 on an EMPTY prefix, which under set -e kills this script
# before it prints anything. The `|| true` goes INSIDE the pipe.
CH=$( { aws s3 ls s3://raglab-zisan-apne1/chunks/ "${P[@]}" 2>/dev/null || true; } | grep -c jsonl.gz || true )
echo "ami=$AMI  type=$TYPE  az=$AZ  SPOT \$$PRICE/hr  job=$JOB  arm=${ARM:-<from job>}  shutdown=$SHUTDOWN  chunk files=$CH"

# JOB is passed through user-data by prepending an export, so one bootstrap file
# serves both the benchmark and the real run.
# GIT_SHA comes from HERE, not from the instance - the box has no repo, so
# resolving it there silently wrote null and lost the provenance of the vectors.
UD=$(mktemp); {
  echo "#!/bin/bash"
  echo "export JOB=$JOB ARM='$ARM' MODEL='$MODEL' SHUTDOWN=$SHUTDOWN LIFETIME=$LIFETIME ALLOW_CPU=$ALLOW_CPU"
  echo "export GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null)"
  tail -n +2 infra/bootstrap_embed.sh
} > "$UD"

ID=$(aws ec2 run-instances "${P[@]}" \
  --image-id "$AMI" --instance-type "$TYPE" \
  --subnet-id "$SUBNET" --security-group-ids "$SG" \
  --iam-instance-profile Name=raglab-ec2-parser \
  --instance-initiated-shutdown-behavior terminate \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}' \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":'"$VOL"',"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data "file://$UD" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=raglab-phase7-embed},{Key=Project,Value=raglab}]' \
  --query 'Instances[0].InstanceId' --output text)
rm -f "$UD"

echo
echo "launched: $ID"
echo "  log:    aws s3 cp s3://raglab-zisan-apne1/logs/$ID.log - --profile zisan | tail -40"
echo "  poke:   $PWD/infra/ssm.sh run $ID 'tail -40 /opt/raglab/bootstrap.log'"
echo "  kill:   $PWD/infra/ssm.sh kill $ID"
