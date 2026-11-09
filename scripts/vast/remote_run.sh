#!/usr/bin/env bash
# Runs on the Vast instance; started by the onstart command from scripts/vast/launch.py.
# Env: HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3, RUN_ID, S3_PREFIX
set -uo pipefail
cd /root/lexsi
export PYTHONPATH=/root/lexsi/src PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1 PIP_BREAK_SYSTEM_PACKAGES=1
PY=python3
stamp() { date -u +%FT%TZ; }
mark() { echo "$1" > STATUS; $PY -m lfm3u.storage put STATUS "$S3_PREFIX/STATUS" >/dev/null 2>&1 || true; }

echo "== $(stamp) setup on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DEPS='math-verify[antlr4_13_2] rouge-score boto3 pyarrow'
if ! $PY -m pip install -q --no-cache-dir $DEPS; then
  uv pip install --system -q $DEPS || { echo "dependency install failed"; exit 1; }
fi
mark RUNNING
( while true; do sleep 30; $PY -m lfm3u.storage put run.log "$S3_PREFIX/run.log" >/dev/null 2>&1; done ) &
SYNC=$!
$PY -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
$PY -c "import vllm; print('vllm', vllm.__version__)" || echo "vllm import failed"

# FlashInfer autotune + torch.compile caches, keyed by GPU model (saves ~7 min of engine warmup per run)
GPU_KEY=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_')
CACHE_URI="$(echo "$S3_PREFIX" | cut -d/ -f1-3)/cache/vllm-cf5cea2-$GPU_KEY.tgz"
if $PY -m lfm3u.storage get "$CACHE_URI" /tmp/cache.tgz >/dev/null 2>&1; then
  tar xzf /tmp/cache.tgz -C / && echo "restored compile cache $CACHE_URI"
fi

echo "== $(stamp) eval ${EVAL_ARGS:-}"
$PY scripts/01_baseline_eval.py --out results/base --run-id "$RUN_ID" ${EVAL_ARGS:-}
RC=$?
echo "== $(stamp) eval exit $RC"
$PY -m lfm3u.storage upload results/base "$S3_PREFIX/results"
if tar czf /tmp/cache.tgz --ignore-failed-read -C / root/.cache/flashinfer root/.cache/vllm 2>/dev/null; then
  $PY -m lfm3u.storage put /tmp/cache.tgz "$CACHE_URI" && echo "saved compile cache $CACHE_URI"
fi
kill $SYNC 2>/dev/null
$PY -m lfm3u.storage put run.log "$S3_PREFIX/run.log"
if [ "$RC" -eq 0 ]; then mark DONE; else mark "FAILED rc=$RC"; fi
