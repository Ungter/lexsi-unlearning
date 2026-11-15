#!/usr/bin/env bash
# Fine-tune ONE MUSE model with the general-text anchor, upload it, and run the full baseline eval on it.
#   bash scripts/vast/jobs/refit_one.sh CORPUS TAG [extra scripts/03_finetune.py args...]
# CORPUS is target or retain. Outputs: s3://<bucket>/models/$RUN_ID/aya-muse-CORPUS-TAG/, results/finetune/CORPUS-TAG.json,
# results/CORPUS_TAG/ (baseline.json + generations). Running target and retain on separate single-GPU boxes in parallel
# gives the same speed-up as a multi-GPU box at single-GPU prices.
set -uo pipefail
source scripts/vast/common.sh
CORPUS=${1:?corpus (target|retain)}; TAG=${2:?tag}; shift 2
# Replay rows are token ids: the LFM3 replay (runs/ft-20261108-233952) cannot be reused for Tiny Aya. Set REPLAY_URI to
# an Aya replay file from an earlier run; otherwise it is generated here with the base model and uploaded to
# $S3_PREFIX/work/replay.jsonl so the other corpus's box can reuse it. The FineWeb anchor is plain text (reusable).
REPLAY_URI=${REPLAY_URI:-}
ANCHOR_URI=${ANCHOR_URI:-$BUCKET_URI/eval/fineweb_anchor.jsonl}
NAME=aya-muse-$CORPUS-$TAG
mkdir -p work models results/finetune

stage "inputs: replay + anchor"
if [ -n "$REPLAY_URI" ]; then
  $PY -m lfm3u.storage get "$REPLAY_URI" work/replay.jsonl || fail "replay download"
else
  stage "self-replay generation (base Tiny Aya)"
  $PY scripts/02_replay.py --out work/replay.jsonl || fail "replay"
  cp work/replay.jsonl.summary.json results/finetune/replay_summary.json 2>/dev/null
  $PY -m lfm3u.storage put work/replay.jsonl "$S3_PREFIX/work/replay.jsonl" || true
  $PY -m lfm3u.storage put work/replay.jsonl.summary.json "$S3_PREFIX/work/replay.jsonl.summary.json" || true
fi
$PY -m lfm3u.storage get "$ANCHOR_URI" work/fineweb_anchor.jsonl || fail "anchor download"

stage "fine-tune $CORPUS ($TAG) $*"
$PY scripts/03_finetune.py --corpus "$CORPUS" --replay work/replay.jsonl --anchor work/fineweb_anchor.jsonl \
  --out "models/$NAME" --log "results/finetune/$CORPUS-$TAG.json" "$@" || fail "finetune $CORPUS $TAG"
sync_up

stage "upload $NAME"
upload_model "models/$NAME" "$NAME" || fail "upload $NAME"

stage "eval $NAME"
$PY scripts/01_baseline_eval.py --model "models/$NAME" --out "results/${CORPUS}_$TAG" --run-id "$RUN_ID" \
  ${EVAL_ARGS:-} || fail "eval $NAME"
