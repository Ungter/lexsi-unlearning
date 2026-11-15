#!/usr/bin/env bash
# Dev-split self-generated solutions from the reference target: data for the in-context and relearning attacks (PLAN
# 6.3 #9, #10). Writes $SELFGEN_DEV_URI (default s3://<bucket>/aya/unlearn/selfgen_dev.jsonl, the path
# attack_models.sh / unlearn_relearn.sh read; calculus + algebra/precalculus dev, direct mode; Tiny Aya has no thinking
# mode). The LFM3 file unlearn/selfgen_dev_v3.jsonl holds LFM3 token ids and is left untouched. TARGET_URI must point at
# the Aya reference target (aya-muse-target-*).
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}  # any LFM3-built book input_ids are re-tokenized from text
TARGET_URI=${TARGET_URI:?set TARGET_URI to the Aya reference target, s3://.../models/<run>/aya-muse-target-<tag>}
SELFGEN_DEV_URI=${SELFGEN_DEV_URI:-$BUCKET_URI/aya/unlearn/selfgen_dev.jsonl}
mkdir -p work models results/unlearn
{ $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
TARGET=$(fetch_model "$TARGET_URI" models/target) || fail "target"
stage "selfgen dev"
$PY scripts/06_selfgen.py --model "$TARGET" --split dev --calc-per-level 60 --think-frac 0 --retain-per-level 40 \
  --retain-topics algebra,precalculus --general 0 --out work/selfgen_dev.jsonl || fail "selfgen"
$PY -m lfm3u.storage put work/selfgen_dev.jsonl "$SELFGEN_DEV_URI" || fail "upload"
cp work/selfgen_dev.jsonl.summary.json results/unlearn/selfgen_dev_summary.json
stage "done"
