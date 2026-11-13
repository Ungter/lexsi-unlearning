#!/usr/bin/env bash
# Paper experiment P5 (relearning-aware weight surgery): one box for the l2p5_* sessions, driven over SSH.
#   1. dead-man timer: the instance destroys itself MAX_HOURS + 0.25 h after this script starts, whatever happens
#   2. peft (the relearning probe is the LoRA of scripts/12_relearn.py), torch/transformers must not change
#   3. /root/lexsi/.box_env: the container environment for SSH sessions (credentials stay on the box, mode 600)
#   4. the Phase-7 worker setup (jlens, data, target, bundle, selftest) and its S3 command queue as a fallback;
#      the worker ends after MAX_HOURS, and without --keep the instance then destroys itself (remote_run.sh)
#   args: MAX_HOURS (default 13.5; keep MAX_HOURS + 0.25 <= 14)
set -uo pipefail
source scripts/vast/common.sh
MAX_HOURS=${1:-13.5}
DEAD_S=$($PY -c "print(int(($MAX_HOURS + 0.25) * 3600))")

stage "p5: dead-man timer, self-destroy in $DEAD_S s"
if [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
  setsid nohup bash -c "sleep $DEAD_S; for n in 1 2 3 4 5 6 7 8; do vastai destroy instance $CONTAINER_ID -y --api-key $CONTAINER_API_KEY && break; sleep 30; done" \
    > /root/deadman.log 2>&1 < /dev/null &
  echo "dead-man pid $! (instance $CONTAINER_ID), fires at $(date -u -d "+$DEAD_S seconds" +%FT%TZ)"
else
  fail "CONTAINER_ID / CONTAINER_API_KEY not set: no dead-man timer, refusing to run"
fi

stage "p5: peft"
before=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
$PY -m pip install -q --no-cache-dir peft || fail "pip install peft"
after=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
[ "$before" = "$after" ] || fail "peft changed torch/transformers ($before -> $after)"
$PY -c "import peft; print('peft', peft.__version__)"

stage "p5: env file for SSH sessions"
( umask 077; export -p | grep -E '^declare -x (AWS_|HF_|S3_PREFIX|RUN_ID|CODE_SHA|CONTAINER_ID|CONTAINER_API_KEY|PYTHONPATH|PYTHONUNBUFFERED|HF_XET|PIP_BREAK)' > /root/lexsi/.box_env )
echo "wrote .box_env ($(wc -l < /root/lexsi/.box_env) variables)"
mkdir -p steps_in

bash scripts/vast/jobs/surgery_worker.sh "$MAX_HOURS" "$MAX_HOURS" || fail "surgery worker"
