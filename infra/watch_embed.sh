#!/bin/bash
# LIVE progress for the Phase 7 embedding job.
# ---------------------------------------------------------------------------
#   infra/watch_embed.sh <instance-id>
#   watch -n 30 'infra/watch_embed.sh <instance-id>'
#   infra/watch_embed.sh <instance-id> --s3     # if the SSM agent is down
#
# WHY IT LOOKS LIKE THIS:
#   * It reads the log from the INSTANCE over SSM, not from S3. The first
#     version read S3, which the box pushes on a 60s timer, and - worse - fell
#     back to <id>-resume.log, a log that had finished hours earlier. It showed
#     a frozen, finished log while the real job ran invisibly. A watcher that
#     presents stale output as current is worse than no watcher.
#   * SHARD COUNTS COME FROM S3 AND ARE GLOBAL. With two boxes running different
#     arms, pointing this at box 2 still showed box 1's arm filling up, which
#     reads as "wrong box". So the arm THIS box owns is read off its user-data
#     and marked, and the header says the counts cover both boxes.
#   * It names the boot phase. Between launch and the first shard there are
#     several minutes of pip install and model restore during which the GPU sits
#     at 0% and the log is empty - which looks identical to a dead job.
#
# WHAT IT STILL CANNOT DO: show the spot 2-minute interruption notice. That is
# delivered to the instance at 169.254.169.254 and is invisible from outside.
# The poller inside embed.py is the only thing that can see it.
# ---------------------------------------------------------------------------
set -uo pipefail
ID=${1:?usage: watch_embed.sh <instance-id> [--s3]}
MODE=${2:-ssm}
P=(--profile zisan --region ap-northeast-1)
B=s3://raglab-zisan-apne1
MODEL=Qwen3-Embedding-0.6B

read -r ST AZ < <(aws ec2 describe-instances "${P[@]}" --instance-ids "$ID" \
  --query 'Reservations[0].Instances[0].[State.Name,Placement.AvailabilityZone]' --output text 2>/dev/null)

MINE=""; LIVE=""
if [ "$MODE" != "--s3" ] && [ "$ST" = "running" ]; then
  JF=$(mktemp)
  cat > "$JF" <<'JSON'
{"commands":["grep -o \"ARM='[^']*'\" /var/lib/cloud/instance/user-data.txt 2>/dev/null | head -1; echo '@@'; nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader; echo '@@'; pgrep -af 'embed.py' | head -1; echo '@@'; tail -4 /opt/raglab/bootstrap.log 2>/dev/null | tr -d '\\r' | tail -1; echo '@@'; cat /opt/raglab/embed.log /opt/raglab/bootstrap.log 2>/dev/null | grep -vE 'Loading weights|^[[:space:]]*$' | grep -E 'shard 000|arm=|FAILED|SPOT INTERRUPTION|COMPLETE|INCOMPLETE' | tail -12"]}
JSON
  CID=$(aws ssm send-command "${P[@]}" --instance-ids "$ID" --document-name AWS-RunShellScript \
          --parameters "file://$JF" --query 'Command.CommandId' --output text 2>/dev/null)
  rm -f "$JF"
  if [ -n "${CID:-}" ]; then
    for _ in $(seq 1 12); do
      sleep 1
      S=$(aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" \
            --query Status --output text 2>/dev/null || echo Pending)
      [ "$S" = "InProgress" ] || [ "$S" = "Pending" ] || break
    done
    LIVE=$(aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" \
             --query StandardOutputContent --output text 2>/dev/null)
    MINE=$(echo "$LIVE" | awk -F"'" '/^ARM=/{print $2; exit}')
  fi
fi

sec() { echo "$LIVE" | awk -v n="$1" 'BEGIN{i=0} /^@@$/{i++; next} i==n'; }
GPU=$(sec 1); PROC=$(sec 2); BOOT=$(sec 3); JOBLOG=$(sec 4)

printf "instance %s  state=%s  az=%s   %s\n" "$ID" "${ST:-?}" "${AZ:-?}" "$(date -u '+%H:%M:%SZ')"
printf "this box owns: %s\n\n" "${MINE:-<unknown>}"

echo "shards in S3 (GLOBAL - covers every box working this job):"
for pair in "B1_3200_ov889:19" "A_2400_ov360:50"; do
  SLUG=${pair%%:*}; N=${pair##*:}
  C=$( { aws s3 ls "$B/vectors/$SLUG/$MODEL/" "${P[@]}" 2>/dev/null || true; } | grep -c '\.npy$' )
  [ "$SLUG" = "$MINE" ] && MARK="*" || MARK=" "
  printf " %s %-16s %2d/%-2d  [" "$MARK" "$SLUG" "$C" "$N"
  for i in $(seq 1 25); do [ $((i*N)) -le $((C*25)) ] && printf "#" || printf "."; done
  printf "]\n"
done
E=$( { aws s3 ls "$B/errors/embed/" --recursive "${P[@]}" 2>/dev/null || true; } | grep -c '\.json$' )
printf "   %-16s errors=%d        (* = this box)\n\n" "" "$E"

if [ "$MODE" = "--s3" ]; then
  echo "--- log from S3 (up to 60s stale) ---"
  aws s3 cp "$B/logs/$ID-embed.log" - "${P[@]}" 2>/dev/null \
    | grep -vE 'Loading weights|^\s*$' | tail -12 || echo "  (no embed log in S3 yet)"
  exit 0
fi

printf "gpu: %s\n" "${GPU:-unreachable}"
if [ -n "$PROC" ]; then
  echo "phase: EMBEDDING   ($PROC)"
  echo; echo "--- embed.log (live) ---"; echo "$JOBLOG"
else
  echo "phase: STILL SETTING UP - pip install + 3.1GB model restore, several minutes."
  echo "       GPU at 0% here is expected, not a dead job."
  echo; echo "--- bootstrap.log (live, last line) ---"; echo "${BOOT:-...}"
fi
