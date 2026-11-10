#!/usr/bin/env bash
# Dev-split self-generated solutions from the reference target: data for the in-context and relearning attacks (PLAN
# 6.3 #9, #10). Writes s3://<bucket>/unlearn/selfgen_dev_v3.jsonl (calculus + algebra/precalculus dev, direct mode).
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=$BUCKET_URI/data/phase2-20261109.tgz
TARGET_URI=$BUCKET_URI/models/script-20261109-044309/lfm3-muse-target-A
mkdir -p work models results/unlearn
{ $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
TARGET=$(fetch_model "$TARGET_URI" models/target) || fail "target"
stage "selfgen dev"
$PY scripts/06_selfgen.py --model "$TARGET" --split dev --calc-per-level 60 --think-frac 0 --retain-per-level 40 \
  --retain-topics algebra,precalculus --general 0 --out work/selfgen_dev.jsonl || fail "selfgen"
$PY -m lfm3u.storage put work/selfgen_dev.jsonl "$BUCKET_URI/unlearn/selfgen_dev_v3.jsonl" || fail "upload"
cp work/selfgen_dev.jsonl.summary.json results/unlearn/selfgen_dev_summary.json
stage "done"
