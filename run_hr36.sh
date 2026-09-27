#!/bin/bash
# Horns Rev at 36 wind directions (WFLOP_HR_DIRS=36).
# Usage: ./run_hr36.sh "<ALG:wake> [...]" <workers per job> <result-branch>
# Jobs run side by side; the results are pushed every 10 minutes and at the end.
set -u; cd "$(dirname "$0")"; JOBS="$1"; WORKERS="$2"; BRANCH="$3"
git config user.name prince197; git config user.email 93870989+prince197@users.noreply.github.com
git checkout -q -B "$BRANCH"
push() {
  git add -f results curves logs log36_*.txt run_status.txt 2>/dev/null
  git commit -q -m "Horns Rev 36-direction results ($JOBS): $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HsBwe7Pwhk6cRCQhisdU8J" 2>/dev/null
  for i in 1 2 3 4 5; do git push -q -f -u origin "$BRANCH" && return 0; sleep $((2**i)); done
}
PIDS=""
for spec in $JOBS; do ALG=${spec%%:*}; WAKE=${spec##*:}
  ( WFLOP_SITE=hornsrev WFLOP_HR_DIRS=36 WFLOP_ALGOS=$ALG WFLOP_WAKE=$WAKE WFLOP_TAG=d36_${WAKE}_${ALG} \
    WFLOP_BACKEND=cpu WFLOP_WORKERS=$WORKERS python -u run_experiments_gpu.py > log36_${WAKE}_${ALG}.txt 2>&1
    echo "$WAKE $ALG exit $?" >> run_status.txt ) &
  PIDS="$PIDS $!"
done
running() { for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
while running; do
  for _ in $(seq 60); do running || break; sleep 10; done
  running && push "checkpoint $(date -u +%H:%M)"
done
push "final"; echo PUSHED >> run_status.txt
