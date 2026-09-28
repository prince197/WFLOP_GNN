#!/bin/bash
# GNN-LX-SSA at Horns Rev, 36 directions, split into shards that run side by side.
# Usage: ./run_hr36_shards.sh <wake> "<shard ids>" <runs per shard> <result-branch>
# Each shard k runs <runs per shard> runs from its own seed (WFLOP_SHARD=k). Results are
# pushed every 10 minutes and at the end.
set -u; cd "$(dirname "$0")"; WAKE="$1"; SHARDS="$2"; RUNS="$3"; BRANCH="$4"
git config user.name prince197; git config user.email 93870989+prince197@users.noreply.github.com
git checkout -q -B "$BRANCH"
push() {
  git add -f results curves logs log36_*.txt run_status.txt 2>/dev/null
  git commit -q -m "Horns Rev 36-direction GNN-LX-SSA shards ($WAKE: $SHARDS): $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HsBwe7Pwhk6cRCQhisdU8J" 2>/dev/null
  for i in 1 2 3 4 5; do git push -q -f -u origin "$BRANCH" && return 0; sleep $((2**i)); done
}
PIDS=""
for k in $SHARDS; do
  ( WFLOP_SITE=hornsrev WFLOP_HR_DIRS=36 WFLOP_ALGOS=GNNLXSSA WFLOP_WAKE=$WAKE WFLOP_RUNS=$RUNS WFLOP_SHARD=$k \
    WFLOP_TAG=d36_${WAKE}_GNNLXSSA_s${k} WFLOP_BACKEND=cpu WFLOP_WORKERS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python -u run_experiments_gpu.py > log36_${WAKE}_GNNLXSSA_s${k}.txt 2>&1
    echo "$WAKE GNNLXSSA shard $k exit $?" >> run_status.txt ) &
  PIDS="$PIDS $!"
done
running() { for p in $PIDS; do kill -0 $p 2>/dev/null && return 0; done; return 1; }
while running; do
  for _ in $(seq 60); do running || break; sleep 10; done
  running && push "checkpoint $(date -u +%H:%M)"
done
push "final"; echo PUSHED >> run_status.txt
