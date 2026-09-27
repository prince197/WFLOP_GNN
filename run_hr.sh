#!/bin/bash
# Horns Rev campaign worker. Usage: ./run_hr.sh "<ALG:wake> [<ALG:wake> ...]" <workers> <result-branch>
# Runs each job in parallel, then commits results/, curves/, logs and pushes them to <result-branch>.
set -u
cd "$(dirname "$0")"
JOBS="$1"; WORKERS="$2"; BRANCH="$3"
for spec in $JOBS; do
  ALG=${spec%%:*}; WAKE=${spec##*:}
  ( WFLOP_SITE=hornsrev WFLOP_ALGOS=$ALG WFLOP_WAKE=$WAKE WFLOP_TAG=${WAKE}_${ALG} WFLOP_BACKEND=cpu \
    WFLOP_WORKERS=$WORKERS python -u run_experiments_gpu.py > log_${WAKE}_${ALG}.txt 2>&1
    echo "$WAKE $ALG exit $?" >> run_status.txt ) &
done
wait
git config user.name prince197
git config user.email 93870989+prince197@users.noreply.github.com
git checkout -B "$BRANCH"
git add -f results curves logs log_*.txt run_status.txt 2>/dev/null
git commit -q -m "Horns Rev campaign results: $JOBS

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HsBwe7Pwhk6cRCQhisdU8J"
for i in 1 2 3 4 5; do git push -u origin "$BRANCH" && break; sleep $((2**i)); done
echo PUSHED >> run_status.txt
