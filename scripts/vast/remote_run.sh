#!/usr/bin/env bash
# Runs on the Vast instance; started by the onstart command from scripts/vast/launch.py.
# Env: HF_TOKEN, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3, RUN_ID, S3_PREFIX, CODE_SHA,
#      JOB: one of the jobs below, or several joined with "+" (run in order)
#        baseline  scripts/01_baseline_eval.py on LFM3-1.7B, or on MODEL_URI -> results/EVAL_NAME
#        eval      scripts/01_baseline_eval.py on each EVAL_MODELS entry ("name=ref,name=ref"; ref = s3:// prefix or
#                  HF id) -> results/<name>
#        finetune  self-replay (or REPLAY_URI), fine-tune target + retain-only (FT_ARGS), upload, evaluate
#        script    run each JOB_SCRIPT (comma list of repo paths) with SCRIPT_ARGS; scripts may
#                  "source scripts/vast/common.sh" for stage/fail/sync_up/fetch_model/upload_model
#      EVAL_ARGS apply to every 01_baseline_eval.py call.
# Everything under results/ is mirrored to $S3_PREFIX/results; checkpoints go to s3://<bucket>/models/$RUN_ID/.
set -uo pipefail
cd /root/lexsi
source scripts/vast/common.sh
JOB=${JOB:-baseline}
mark() { echo "$1" > STATUS; $PY -m lfm3u.storage put STATUS "$S3_PREFIX/STATUS" >/dev/null 2>&1 || true; }
# AUTO_DESTROY=1 (launch.py default): the instance destroys itself with its per-instance key once results are synced
self_destroy() {
  if [ "${AUTO_DESTROY:-0}" = "1" ] && [ -n "${CONTAINER_ID:-}" ] && [ -n "${CONTAINER_API_KEY:-}" ]; then
    echo "== $(stamp) self-destroy instance $CONTAINER_ID"
    $PY -m lfm3u.storage put run.log "$S3_PREFIX/run.log" >/dev/null 2>&1
    vastai destroy instance "$CONTAINER_ID" -y --api-key "$CONTAINER_API_KEY" \
      || echo y | vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
  elif [ "${AUTO_DESTROY:-0}" = "1" ]; then
    echo "== self-destroy skipped: CONTAINER_ID/CONTAINER_API_KEY not set"
  fi
}
fail() { echo "== $(stamp) FAILED: $1"; kill "${SYNC:-0}" 2>/dev/null; sync_up; mark "FAILED $1"; self_destroy; exit 1; }
cache_sig() { find /root/.cache/vllm /root/.cache/flashinfer -type f -printf '%s %T@\n' 2>/dev/null | md5sum | cut -c1-32; }

echo "== $(stamp) setup on $(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
DEPS='math-verify[antlr4_13_2] rouge-score boto3 pyarrow langdetect immutabledict nltk vastai'
if ! $PY -m pip install -q --no-cache-dir $DEPS; then
  uv pip install --system -q $DEPS || { echo "dependency install failed"; exit 1; }
fi
# IFEval checkers only (lfm3u.audit.utility imports them without the rest of the harness)
$PY -m pip install -q --no-cache-dir --no-deps lm-eval || echo "lm-eval install failed (ifeval tasks will fail)"
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
CACHE_BEFORE=$(cache_sig)

mkdir -p results
for J in ${JOB//+/ }; do
case "$J" in
  baseline)
    MODEL=() NAME=base
    if [ -n "${MODEL_URI:-}" ]; then
      stage "download $MODEL_URI"
      $PY -m lfm3u.storage download "$MODEL_URI" models/eval || fail download
      MODEL=(--model models/eval) NAME=${EVAL_NAME:-eval}
    fi
    stage "eval $NAME ${EVAL_ARGS:-}"
    $PY scripts/01_baseline_eval.py --out "results/$NAME" --run-id "$RUN_ID" "${MODEL[@]}" ${EVAL_ARGS:-} || fail "eval $NAME"
    ;;
  eval)
    IFS=',' read -ra ENTRIES <<< "${EVAL_MODELS:?EVAL_MODELS is required for JOB=eval}"
    for E in "${ENTRIES[@]}"; do
    NAME=${E%%=*} REF=${E#*=}
    stage "eval $NAME ($REF) ${EVAL_ARGS:-}"
    MP=$(fetch_model "$REF" "models/$NAME") || fail "download $REF"
    $PY scripts/01_baseline_eval.py --model "$MP" --out "results/$NAME" --run-id "$RUN_ID" ${EVAL_ARGS:-} || fail "eval $NAME"
    sync_up
    done
    ;;
  script)
    IFS=',' read -ra SCRIPTS <<< "${JOB_SCRIPT:?JOB_SCRIPT is required for JOB=script}"
    for S in "${SCRIPTS[@]}"; do
    stage "script $S ${SCRIPT_ARGS:-}"
    bash "$S" ${SCRIPT_ARGS:-} || fail "script $S"
    sync_up
    done
    ;;
  finetune)
    mkdir -p work models results/finetune
    stage "smoke: 2 training steps on the real model"
    $PY scripts/03_finetune.py --corpus target --max-steps 2 --no-save --no-probes --epochs 1 || fail smoke
    if [ -n "${REPLAY_URI:-}" ]; then
      stage "reuse self-replay $REPLAY_URI"
      $PY -m lfm3u.storage get "$REPLAY_URI" work/replay.jsonl || fail "replay download"
      $PY -m lfm3u.storage get "$REPLAY_URI.summary.json" work/replay.jsonl.summary.json || true
    else
      stage "self-replay generation"
      $PY scripts/02_replay.py --out work/replay.jsonl || fail replay
    fi
    cp work/replay.jsonl.summary.json results/finetune/replay_summary.json 2>/dev/null
    $PY -m lfm3u.storage put work/replay.jsonl "$S3_PREFIX/work/replay.jsonl" || true
    $PY -m lfm3u.storage put work/replay.jsonl.summary.json "$S3_PREFIX/work/replay.jsonl.summary.json" || true
    for C in target retain; do
      stage "fine-tune $C ${FT_ARGS:-}"
      $PY scripts/03_finetune.py --corpus $C --replay work/replay.jsonl --out "models/lfm3-muse-$C" \
        --log "results/finetune/$C.json" ${FT_ARGS:-} || fail "finetune $C"
      sync_up
      stage "upload models/lfm3-muse-$C"
      $PY -m lfm3u.storage upload "models/lfm3-muse-$C" "$BUCKET_URI/models/$RUN_ID/lfm3-muse-$C" || fail "upload $C"
    done
    for C in target retain; do
      stage "eval $C"
      $PY scripts/01_baseline_eval.py --model "models/lfm3-muse-$C" --out "results/$C" --run-id "$RUN_ID" \
        ${EVAL_ARGS:-} || fail "eval $C"
      sync_up
    done
    ;;
  *) fail "unknown JOB=$J" ;;
esac
done

stage "done"
if [ "$(cache_sig)" = "$CACHE_BEFORE" ]; then
  echo "compile cache unchanged (no new vLLM compilation); not re-uploading"
elif tar czf /tmp/cache.tgz --ignore-failed-read -C / root/.cache/flashinfer root/.cache/vllm 2>/dev/null; then
  $PY -m lfm3u.storage put /tmp/cache.tgz "$CACHE_URI" && echo "saved compile cache $CACHE_URI"
fi
kill $SYNC 2>/dev/null
sync_up
mark DONE
self_destroy
