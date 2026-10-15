#!/bin/bash
# Launch the Phase 1 parse job. Prints the instance id and the watch command.
# ---------------------------------------------------------------------------
# REUSABLE. See infra/README.md.
#
# CHANGE FOR A NEW PROJECT: AMI, SG, SUBNET, instance profile name, bucket,
#   default instance type. Resolve the AMI from an SSM public parameter rather
#   than pinning an id - a pinned AMI ages out of patch coverage and is
#   region-specific.
#
# WHY IT LOOKS LIKE THIS:
#   * The banner prints before any fallible command. This script once exited
#     silently with status 1 and printed nothing at all, because...
#   * `aws s3 ls` EXITS 1 ON AN EMPTY PREFIX. Under `set -euo pipefail` that kills
#     the script instantly. Inside `if aws s3 ls ...; then` it is safe (set -e does
#     not fire in a condition); bare in a $( ) it is not.
#   * The `|| true` goes INSIDE the pipe: `{ cmd || true; } | wc -l`. Written as
#     `cmd | wc -l || echo 0` you get "0\n0" - wc prints one and echo prints another.
#   * No `--max-items` on describe-spot-price-history: the CLI appends the
#     pagination token to --output text, leaking a literal "None" into the value.
#   * Instance profile, never access keys. One-time spot + terminate-on-shutdown.
#   * Safe to re-run: the worker skips units that already have output in S3.
# ---------------------------------------------------------------------------
#
#   infra/launch.sh                       # m7i.4xlarge spot, AZ 1d
#   infra/launch.sh --on-demand           # no interruption risk, ~3x the hourly rate
#   infra/launch.sh --az 1a                # different capacity pool after a spot reclaim
#   infra/launch.sh --on-demand m7i.2xlarge
#
# Safe to re-run after an interruption: the worker skips documents that already
# have a parsed/<stem>.json in S3, so a second launch resumes where the last
# one stopped.

set -euo pipefail
cd "$(dirname "$0")/.."
# Say something immediately, so a failure further down is never silent.
echo "raglab phase 1 launcher"
P=(--profile zisan --region ap-northeast-1)

# A spot reclaim ("instance-terminated-no-capacity") is a property of ONE
# capacity pool = one AZ + one instance type. Retrying in a different AZ is
# usually enough; --on-demand removes the risk entirely at ~3x the hourly rate.
SPOT=true; AZ=1d; TYPE=m7i.4xlarge
while [ $# -gt 0 ]; do
  case "$1" in
    --on-demand) SPOT=false; shift ;;
    --az)        AZ=$2; shift 2 ;;
    *)           TYPE=$1; shift ;;
  esac
done

AMI=ami-0739e5f3b35c608e5
SG=sg-07468a8f471d43f07
case "$AZ" in
  1a) SUBNET=subnet-0854039f1813e07ab ;;
  1c) SUBNET=subnet-051959a17410f548f ;;
  1d) SUBNET=subnet-09e0868af873fc031 ;;
  *)  echo "unknown AZ: $AZ (use 1a, 1c or 1d)"; exit 1 ;;
esac

# `aws s3 ls` exits 1 on an EMPTY prefix - which, under `set -e`, kills this
# script before it prints anything. Guard every count and lookup.
# Put `|| true` on the aws call INSIDE the pipe, not after it: `cmd | wc -l || echo 0`
# makes wc print 0 and then echo print 0 again, giving a two-line value.
DONE=$( { aws s3 ls s3://raglab-zisan-apne1/parsed/ "${P[@]}" 2>/dev/null || true; } | wc -l )
PRICE=$(aws ec2 describe-spot-price-history "${P[@]}" --instance-types "$TYPE" \
          --product-descriptions "Linux/UNIX" \
          --query 'SpotPriceHistory[?AvailabilityZone==`ap-northeast-1d`]|[0].SpotPrice' \
          --output text 2>/dev/null | head -1 || echo "?")
if $SPOT; then
  MARKET=(--instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
  echo "type=$TYPE  az=$AZ  SPOT \$$PRICE/hr  already parsed=$DONE/196"
else
  MARKET=()
  echo "type=$TYPE  az=$AZ  ON-DEMAND (spot would be \$$PRICE/hr)  already parsed=$DONE/196"
fi

ID=$(aws ec2 run-instances "${P[@]}" \
  --image-id "$AMI" --instance-type "$TYPE" \
  --subnet-id "$SUBNET" --security-group-ids "$SG" \
  --iam-instance-profile Name=raglab-ec2-parser \
  --instance-initiated-shutdown-behavior terminate \
  "${MARKET[@]}" \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":40,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
  --user-data file://infra/bootstrap.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=raglab-phase1},{Key=Project,Value=raglab}]' \
  --query 'Instances[0].InstanceId' --output text)

echo
echo "launched: $ID"
echo
echo "watch it:"
echo "  watch -n 30 '$PWD/infra/watch.sh $ID'"
echo "kill it:"
echo "  $PWD/infra/ssm.sh kill $ID"
