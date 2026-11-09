#!/usr/bin/env bash
# Phase 3 (J-lens) GPU job, run by remote_run.sh JOB=script from /root/lexsi:
#   python scripts/vast/launch.py up --job script --job-script scripts/vast/jobs/jspace.sh \
#       --script-args "MODELS=base,target,retain FIT_SECONDS=900 RETAIN_FIT_SECONDS=600"
# Args (KEY=VALUE): MODELS (base,target,retain)  FIT_SECONDS (900)  RETAIN_FIT_SECONDS (600)  MIN_PROMPTS (100)
#   MAX_PROMPTS (1000)  DIM_BATCH (auto)  SMOKE (1 = first run every stage on base with a 2-prompt lens; 0 = skip)
#   TARGET_URI / RETAIN_URI (s3:// checkpoint prefixes; default: the ft-20261109-005519 models)
#   QUESTIONS_URI (knowmem jsonl: idx,question,answer,hidden_fact,...; default s3://<bucket>/jspace/inputs/
#     knowmem_forget_qa_jspace.jsonl, whose hidden_fact flags come from the ft-20261109-005519 target)
#   SUFFIX (appended to the output name, e.g. SUFFIX=_v3 -> results/jspace/target_v3, .../jspace/$RUN_ID/target_v3)
# Rerun on new checkpoints (fit + validation + hidden readout, no smoke):
#   --script-args "MODELS=target,retain SMOKE=0 SUFFIX=_v3 TARGET_URI=s3://.../lfm3-muse-target RETAIN_URI=s3://.../lfm3-muse-retain"
# Installs anthropics/jacobian-lens with --no-deps (torch/transformers/numpy must not change), runs a smoke pass,
# then per model: fit (time-budgeted, FineWeb docs 500-1499, 128 tokens) + readouts via scripts/05_jspace.py.
# Small outputs: results/jspace/<model>/ (mirrored to the run prefix). Lens files: s3://<bucket>/jspace/$RUN_ID/<model>/.
set -uo pipefail
source scripts/vast/common.sh

MODELS=base,target,retain FIT_SECONDS=900 RETAIN_FIT_SECONDS=600 MIN_PROMPTS=100 MAX_PROMPTS=1000 DIM_BATCH= SMOKE=1
TARGET_URI=s3://lexsi-hackathon/models/ft-20261109-005519/lfm3-muse-target/
RETAIN_URI=s3://lexsi-hackathon/models/ft-20261109-005519/lfm3-muse-retain/
SUFFIX=
QUESTIONS_URI=s3://lexsi-hackathon/jspace/inputs/knowmem_forget_qa_jspace.jsonl
for kv in "$@"; do
  case "$kv" in
    MODELS=*|FIT_SECONDS=*|RETAIN_FIT_SECONDS=*|MIN_PROMPTS=*|MAX_PROMPTS=*|DIM_BATCH=*|SMOKE=*|TARGET_URI=s3://*|RETAIN_URI=s3://*|SUFFIX=*|QUESTIONS_URI=s3://*)
      eval "${kv%%=*}=\${kv#*=}" ;;
    *) fail "jspace.sh: unknown argument $kv" ;;
  esac
done
echo "jspace: MODELS=$MODELS SUFFIX=$SUFFIX TARGET_URI=$TARGET_URI RETAIN_URI=$RETAIN_URI"
JLENS_SHA=581d398613e5602a5af361e1c34d3a92ea82ba8e  # anthropics/jacobian-lens "Initial release"
LENS_URI="$BUCKET_URI/jspace/$RUN_ID"

stage "jspace: install jlens@${JLENS_SHA:0:7} (--no-deps) + matplotlib"
versions() { $PY -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"; }
BEFORE=$(versions)
$PY -m pip install -q --no-cache-dir --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" \
  || uv pip install --system -q --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" \
  || fail "jlens install"
$PY -c "import matplotlib" 2>/dev/null || $PY -m pip install -q --no-cache-dir matplotlib || echo "matplotlib unavailable: heatmaps skipped"
AFTER=$(versions)
echo "torch/transformers/numpy: $BEFORE -> $AFTER"
[ "$BEFORE" = "$AFTER" ] || fail "an install changed torch/transformers/numpy"
$PY -c "import jlens, huggingface_hub; print('jlens', jlens.__file__)" || fail "jlens import"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

stage "jspace: inputs"
mkdir -p work/jspace models results/jspace
$PY -m lfm3u.storage get s3://lexsi-hackathon/eval/fineweb_sample.jsonl work/fineweb.jsonl || fail "fineweb download"
$PY -m lfm3u.storage get "$QUESTIONS_URI" work/knowmem.jsonl || fail "questions download"
declare -A PIDS
for M in ${MODELS//,/ }; do  # prefetch the fine-tuned checkpoints while the GPU works on base
  case $M in
    target) fetch_model "$TARGET_URI" models/target >/dev/null & PIDS[target]=$! ;;
    retain) fetch_model "$RETAIN_URI" models/retain >/dev/null & PIDS[retain]=$! ;;
  esac
done

COMMON=(--corpus work/fineweb.jsonl --questions work/knowmem.jsonl --min-prompts "$MIN_PROMPTS" --max-prompts "$MAX_PROMPTS")
[ -n "$DIM_BATCH" ] && COMMON+=(--dim-batch "$DIM_BATCH")

if [ "$SMOKE" = 1 ]; then
  stage "jspace: smoke (2-prompt lens on base, every stage)"
  $PY scripts/05_jspace.py --name _smoke --model LiquidAI/LFM3-1.7B "${COMMON[@]}" --max-prompts 2 --min-prompts 1 \
    --chunk 2 --out results/jspace --lens-dir work/smoke || fail "smoke"
  if ls results/jspace/_smoke/*.error.txt >/dev/null 2>&1; then
    cat results/jspace/_smoke/*.error.txt
    fail "smoke stage error"
  fi
  sync_up
fi

for M in ${MODELS//,/ }; do
  FS=$FIT_SECONDS ST=fit,agree,validate,hidden
  case $M in
    base) REF=LiquidAI/LFM3-1.7B SRC=$REF ST=$ST,inject ;;
    target|retain)
      wait "${PIDS[$M]}" || fail "download $M"
      REF=models/$M SRC=$TARGET_URI
      [ "$M" = retain ] && SRC=$RETAIN_URI
      [ -f "$REF/config.json" ] || fail "no checkpoint at $REF"
      [ "$M" = retain ] && FS=$RETAIN_FIT_SECONDS ;;
    *) fail "unknown model $M" ;;
  esac
  N=$M$SUFFIX
  stage "jspace $N ($REF): fit (${FS}s budget, >= $MIN_PROMPTS prompts) + $ST"
  $PY scripts/05_jspace.py --name "$N" --model "$REF" "${COMMON[@]}" --fit-seconds "$FS" --stages "$ST" \
    --source "$SRC" --out results/jspace --lens-dir work/jspace || fail "jspace $N"
  stage "jspace $N: upload lens -> $LENS_URI/$N"
  $PY -m lfm3u.storage upload "work/jspace/$N" "$LENS_URI/$N" || fail "upload lens $N"
  sync_up
done
stage "jspace: done"
