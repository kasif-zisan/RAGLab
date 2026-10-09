#!/bin/bash
# Get at a running raglab instance without SSH keys or open ports.
# ---------------------------------------------------------------------------
# FULLY REUSABLE - nothing in here is project-specific except profile and region.
# See infra/README.md.
#
# WHY IT LOOKS LIKE THIS:
#   * `run` and `peek` use ssm send-command, which is asynchronous (queue, then
#     poll for output) but needs NO local plugin. `shell` uses start-session,
#     which does require session-manager-plugin installed locally. Having both
#     means you can inspect a box before installing anything.
#   * This exists because the instance has ZERO inbound security group rules.
#     SSM reaches it outbound-only, so there is no port 22 and no key pair.
#   * Requires AmazonSSMManagedInstanceCore on the instance role.
# ---------------------------------------------------------------------------
#
#   infra/ssm.sh list                     which instances are reachable
#   infra/ssm.sh peek   <instance-id>     canned status dump (no plugin needed)
#   infra/ssm.sh run    <instance-id> 'command'   run anything (no plugin needed)
#   infra/ssm.sh shell  <instance-id>     interactive shell (needs session-manager-plugin)
#   infra/ssm.sh kill   <instance-id>     terminate it

set -euo pipefail
P=(--profile zisan --region ap-northeast-1)

case "${1:-}" in

list)
  aws ssm describe-instance-information "${P[@]}" \
    --query 'InstanceInformationList[].{Id:InstanceId,Ping:PingStatus,Platform:PlatformName}' --output table
  ;;

run)
  ID=$2; CMD=$3
  # send-command is asynchronous: it queues the command, then you poll for output.
  # It needs no local plugin, which is why it works before you install anything.
  # Build the parameters as real JSON. Interpolating the command into
  # `commands=["$CMD"]` breaks on any nested quote in the command - which is
  # most useful commands. python's json.dumps escapes it correctly.
  JF=$(mktemp)
  CMD="$CMD" python3 -c 'import json,os,sys; json.dump({"commands":[os.environ["CMD"]]}, open(sys.argv[1],"w"))' "$JF"
  CID=$(aws ssm send-command "${P[@]}" --instance-ids "$ID" \
          --document-name AWS-RunShellScript \
          --parameters "file://$JF" \
          --query 'Command.CommandId' --output text)
  rm -f "$JF"
  for _ in $(seq 1 40); do
    sleep 2
    ST=$(aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" \
           --query 'Status' --output text 2>/dev/null || echo Pending)
    [ "$ST" = "InProgress" ] || [ "$ST" = "Pending" ] || break
  done
  aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" \
    --query 'StandardOutputContent' --output text
  aws ssm get-command-invocation "${P[@]}" --command-id "$CID" --instance-id "$ID" \
    --query 'StandardErrorContent' --output text | grep -v '^None$' || true
  ;;

peek)
  ID=$2
  "$0" run "$ID" 'echo \"--- uptime / deadman ---\"; uptime; shutdown --show 2>/dev/null || true; echo; echo \"--- disk ---\"; df -h /opt /; echo; echo \"--- workdir ---\"; ls -la /opt/raglab/ 2>/dev/null; echo; echo \"--- pdfs ---\"; ls /opt/raglab/pdf 2>/dev/null | wc -l; echo; echo \"--- last 25 log lines ---\"; tail -25 /opt/raglab/bootstrap.log 2>/dev/null'
  ;;

shell)
  aws ssm start-session "${P[@]}" --target "$2"
  ;;

kill)
  aws ec2 terminate-instances "${P[@]}" --instance-ids "$2" \
    --query 'TerminatingInstances[0].{Id:InstanceId,State:CurrentState.Name}' --output table
  ;;

*)
  sed -n '2,10p' "$0"; exit 1
  ;;
esac
