#!/bin/bash
# Progress for a running parse job. Reads S3 only - no SSM, no shell, no plugin.
# ---------------------------------------------------------------------------
# REUSABLE. See infra/README.md.
#
# CHANGE FOR A NEW PROJECT: bucket, prefix names, and the hardcoded total (196).
#
# WHY IT LOOKS LIKE THIS:
#   * `set -uo pipefail` WITHOUT -e, deliberately. This script counts S3 objects,
#     and `aws s3 ls` exits 1 on an empty prefix - with -e it would die on an
#     empty output prefix, i.e. exactly at the start of a run.
#   * It reads S3 and nothing else, so it works from any machine with the AWS CLI
#     and needs no SSH key, no SSM plugin, and no inbound access to the instance.
# ---------------------------------------------------------------------------
#
#   infra/watch.sh i-0c5fd89b23ae57021          one snapshot
#   watch -n 30 'infra/watch.sh i-0c5fd89b23ae57021'   live
#
# The log is pushed from the instance every 2 minutes, so it lags by up to that.

set -uo pipefail
ID=${1:?usage: watch.sh <instance-id>}
P=(--profile zisan --region ap-northeast-1)
B=s3://raglab-zisan-apne1

ST=$(aws ec2 describe-instances "${P[@]}" --instance-ids "$ID" \
      --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null)
DONE=$(aws s3 ls $B/parsed/ "${P[@]}" 2>/dev/null | wc -l)
ERRS=$(aws s3 ls $B/errors/ "${P[@]}" 2>/dev/null | wc -l)

printf "instance %s   state=%s\n" "$ID" "$ST"
printf "parsed   %3d / 196   errors %d\n" "$DONE" "$ERRS"
printf "["
for i in $(seq 1 49); do
  [ $((i * 4)) -le "$DONE" ] && printf "#" || printf "."
done
printf "] %d%%\n\n" $((DONE * 100 / 196))

echo "--- last 12 log lines (up to 2 min stale) ---"
aws s3 cp "$B/logs/$ID.log" - "${P[@]}" 2>/dev/null | tail -12 \
  || echo "  (log not in S3 yet - first push is ~2 min after boot)"
