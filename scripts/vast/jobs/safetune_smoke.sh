#!/usr/bin/env bash
# SafeTune smoke test on the MUSE target (scripts/04_safetune_smoke.py); args are passed through.
# Needs the frozen Phase-2 data (s3://lexsi-hackathon/data/phase2-20261109.tgz) and the target checkpoint.
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/data/phase2-20261109.tgz}
TARGET_URI=${TARGET_URI:-$BUCKET_URI/models/ft-20261109-005519/lfm3-muse-target}

stage "install safetune"
before=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
$PY -m pip install -q --no-cache-dir safetune || fail "pip install safetune"
after=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
echo "torch/transformers before: $before | after: $after"
[ "$before" = "$after" ] || echo "WARNING: safetune changed torch/transformers"

stage "data + target model"
if [ ! -f data/MANIFEST.json ]; then
  $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz || fail "data download"
  tar xzf /tmp/data.tgz -C /root/lexsi || fail "data extract"
fi
MP=$(fetch_model "$TARGET_URI" models/target) || fail "target download"

stage "smoke $*"
$PY scripts/04_safetune_smoke.py --model "$MP" --data data --out results/safetune_smoke "$@" || fail smoke
