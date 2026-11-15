#!/usr/bin/env bash
# SafeTune smoke test on the MUSE target (scripts/04_safetune_smoke.py); args are passed through.
# Needs the frozen Phase-2 data (Tiny Aya tokenization, default $BUCKET_URI/aya/data/phase2.tgz) and the target
# checkpoint (TARGET_URI, required: the Aya MUSE target from 03_finetune; the LFM3 lfm3-muse-target must not be used).
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
TARGET_URI=${TARGET_URI:?set TARGET_URI to the Tiny Aya MUSE target checkpoint (s3://.../aya-muse-target)}

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
