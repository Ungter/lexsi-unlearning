#!/usr/bin/env bash
# Fine-tune ONE MUSE model with the general-text anchor, upload it, and run the full baseline eval on it.
#   bash scripts/vast/jobs/refit_one.sh CORPUS TAG [extra scripts/03_finetune.py args...]
# CORPUS is target or retain. Outputs: s3://<bucket>/models/$RUN_ID/lfm3-muse-CORPUS-TAG/, results/finetune/CORPUS-TAG.json,
# results/CORPUS_TAG/ (baseline.json + generations). Running target and retain on separate single-GPU boxes in parallel
# gives the same speed-up as a multi-GPU box at single-GPU prices.
set -uo pipefail
source scripts/vast/common.sh
CORPUS=${1:?corpus (target|retain)}; TAG=${2:?tag}; shift 2
REPLAY_URI=${REPLAY_URI:-$BUCKET_URI/runs/ft-20261108-233952/work/replay.jsonl}
ANCHOR_URI=${ANCHOR_URI:-$BUCKET_URI/eval/fineweb_anchor.jsonl}
NAME=lfm3-muse-$CORPUS-$TAG
mkdir -p work models results/finetune

stage "inputs: replay + anchor"
$PY -m lfm3u.storage get "$REPLAY_URI" work/replay.jsonl || fail "replay download"
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
