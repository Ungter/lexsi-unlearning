#!/usr/bin/env bash
# Build the Phase-7 surgery bundle for Tiny Aya (what scripts/vast/jobs/surgery_worker.sh unpacks) and upload it to
# $BUNDLE_URI (default s3://<bucket>/aya/surgery/bundle-v1.tgz):
#   work/surgery/selfgen_v3.jsonl       <- $SELFGEN_URI      (the target's own TRAIN-split solutions)
#   work/surgery/selfgen_dev_v3.jsonl   <- $SELFGEN_DEV_URI  (DEV-split solutions; the scorer refuses it in sessions)
#   work/surgery/target_v3_lens.pt      <- $LENS_URI/lens.pt (J-lens of the target, scripts/vast/jobs/jspace.sh)
#   results/localize/{book,algebra}/scores.pt <- $LOCALIZE_URI/{book,algebra}/scores.pt
#   work/fineweb_anchor.jsonl           <- s3://<bucket>/eval/fineweb_anchor.jsonl
#   results/surgery/target_ref/dev.json <- 16_surgery.py evalckpt on the target (full preset), computed here
# Each input is waited for (up to WAIT_MIN minutes), so this box can start before the prep / jspace boxes finish.
# Env: TARGET_URI (required), LENS_URI (required, the jspace run's .../aya/jspace/<run>/target), WAIT_MIN (180).
set -uo pipefail
source scripts/vast/common.sh
TARGET_URI=${TARGET_URI:?set TARGET_URI}
LENS_URI=${LENS_URI:?set LENS_URI to s3://.../aya/jspace/<jspace run>/target}
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
SELFGEN_URI=${SELFGEN_URI:-$BUCKET_URI/aya/unlearn/selfgen.jsonl}
SELFGEN_DEV_URI=${SELFGEN_DEV_URI:-$BUCKET_URI/aya/unlearn/selfgen_dev.jsonl}
LOCALIZE_URI=${LOCALIZE_URI:-$BUCKET_URI/aya/localize/v1}
BUNDLE_URI=${BUNDLE_URI:-$BUCKET_URI/aya/surgery/bundle-v1.tgz}
WAIT_MIN=${WAIT_MIN:-180}
mkdir -p work/surgery results/localize/book results/localize/algebra results/surgery models

waitget() {  # waitget URI DEST
  local t=0
  until $PY -m lfm3u.storage get "$1" "$2" >/dev/null 2>&1; do
    [ $t -ge $((WAIT_MIN * 60)) ] && fail "timed out waiting for $1"
    [ $t -eq 0 ] && echo "waiting for $1"
    sleep 60; t=$((t + 60))
  done
  echo "got $1"
}

stage "bundle: data + target"
{ $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
TARGET=$(fetch_model "$TARGET_URI" models/target) || fail "target download"
$PY -m lfm3u.storage get "$BUCKET_URI/eval/fineweb_anchor.jsonl" work/fineweb_anchor.jsonl || fail "anchor"

stage "bundle: selfgen + selfgen_dev"
waitget "$SELFGEN_URI" work/surgery/selfgen_v3.jsonl
waitget "$SELFGEN_DEV_URI" work/surgery/selfgen_dev_v3.jsonl  # the reference's nll_* tasks read the dev file
stage "bundle: target reference (16_surgery.py evalckpt, full preset)"
$PY scripts/16_surgery.py evalckpt "$TARGET" --out results/surgery/target_ref --tasks full || fail "target_ref"
sync_up
stage "bundle: localizer scores, lens"
waitget "$LOCALIZE_URI/book/scores.pt" results/localize/book/scores.pt
waitget "$LOCALIZE_URI/algebra/scores.pt" results/localize/algebra/scores.pt
waitget "$LENS_URI/lens.pt" work/surgery/target_v3_lens.pt

stage "bundle: pack + upload $BUNDLE_URI"
tar czf /tmp/bundle.tgz work/surgery work/fineweb_anchor.jsonl results/localize results/surgery/target_ref || fail "tar"
$PY -m lfm3u.storage put /tmp/bundle.tgz "$BUNDLE_URI" || fail "bundle upload"
ls -la work/surgery results/localize/* results/surgery/target_ref
stage "bundle done"
