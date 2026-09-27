#!/bin/bash
# Commercial power curve campaign (WFLOP_POWER=ge15, Jensen wake).
# Usage: ./run_pc.sh <ALGOS comma list> <dataset 1|2> <cases: all | "500:2,500:3,..."> <workers> <result-branch>
# Resumes from the checkpoint on <result-branch> if an earlier session left one, and pushes
# the checkpoint every 10 minutes so an interrupted session loses at most 10 minutes of work.
set -u; cd "$(dirname "$0")"
ALGOS="$1"; DS="$2"; CASES="$3"; WORKERS="$4"; BRANCH="$5"
git config user.name prince197; git config user.email 93870989+prince197@users.noreply.github.com
git checkout -q -B "$BRANCH"
if git fetch -q origin "$BRANCH" 2>/dev/null; then
  git checkout -q FETCH_HEAD -- results 2>/dev/null && echo "resumed checkpoint from origin/$BRANCH"
fi
push() {
  git add -f results curves logs log_pc.txt run_status.txt 2>/dev/null
  git commit -q -m "Commercial power curve results ($ALGOS, ds$DS): $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HsBwe7Pwhk6cRCQhisdU8J" 2>/dev/null
  for i in 1 2 3 4 5; do git push -q -f -u origin "$BRANCH" && return 0; sleep $((2**i)); done
}
export WFLOP_POWER=ge15 WFLOP_WAKE=jensen WFLOP_ALGOS="$ALGOS" WFLOP_DATASET="$DS" \
       WFLOP_TAG="ge15_${ALGOS//,/-}_$( [ "$CASES" = all ] && echo all || echo "${CASES%%:*}" )" \
       WFLOP_BACKEND=cpu WFLOP_WORKERS="$WORKERS"
[ "$CASES" != all ] && export WFLOP_CASES="$CASES"
python -u run_experiments_gpu.py > log_pc.txt 2>&1 &
PID=$!
while kill -0 $PID 2>/dev/null; do
  for _ in $(seq 60); do kill -0 $PID 2>/dev/null || break; sleep 10; done
  kill -0 $PID 2>/dev/null && push "checkpoint $(date -u +%H:%M)"
done
wait $PID; echo "exit $?" >> run_status.txt
push "final"; echo PUSHED >> run_status.txt
