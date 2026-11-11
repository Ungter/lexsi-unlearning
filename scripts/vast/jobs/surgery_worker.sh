#!/usr/bin/env bash
# Phase 7 worker box: sets up the surgery environment (jlens, frozen data, the MUSE target, the surgery bundle with the
# selfgen files, lens, FineWeb anchor, localizer scores and the unedited target's reference dev scores), runs the
# selftest, then serves commands queued by scripts/vast/surg_remote.py (see scripts/vast/surg_worker.py).
#   args: MAX_HOURS (default 12) IDLE_HOURS (default 2)
set -uo pipefail
source scripts/vast/common.sh
MAX_HOURS=${1:-12}
IDLE_HOURS=${2:-2}
TARGET_URI=${TARGET_URI:-$BUCKET_URI/models/script-20261109-044309/lfm3-muse-target-A}
DATA_URI=${DATA_URI:-$BUCKET_URI/data/phase2-20261109.tgz}
BUNDLE_URI=${BUNDLE_URI:-$BUCKET_URI/surgery/bundle-v1.tgz}
JLENS_SHA=581d398613e5602a5af361e1c34d3a92ea82ba8e  # anthropics/jacobian-lens "Initial release", as in jspace.sh

stage "surgery: install jlens@${JLENS_SHA:0:7} (--no-deps)"
versions() { $PY -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"; }
BEFORE=$(versions)
$PY -m pip install -q --no-cache-dir --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" \
  || uv pip install --system -q --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" \
  || fail "jlens install"
[ "$BEFORE" = "$(versions)" ] || fail "jlens install changed torch/transformers/numpy"
$PY -c "import jlens" || fail "jlens import"

stage "surgery: data, target, bundle"
mkdir -p models results
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
fetch_model "$TARGET_URI" models/target >/dev/null || fail "target download"
{ $PY -m lfm3u.storage get "$BUNDLE_URI" /tmp/bundle.tgz && tar xzf /tmp/bundle.tgz -C /root/lexsi; } || fail "bundle"
ls -la work/surgery results/surgery

stage "surgery: selftest"
if $PY scripts/16_surgery.py selftest --model > results/surgery_selftest.txt 2>&1; then
  echo "selftest passed"
else
  echo "selftest FAILED (results/surgery_selftest.txt)"
fi
tail -20 results/surgery_selftest.txt
sync_up

stage "surgery: worker (max $MAX_HOURS h, idle $IDLE_HOURS h)"
$PY scripts/vast/surg_worker.py "$MAX_HOURS" "$IDLE_HOURS" || fail "worker"
