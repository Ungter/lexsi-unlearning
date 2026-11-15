#!/usr/bin/env bash
# Upload the allowlisted Tiny Aya checkpoints from S3 to Hugging Face: scripts/hf_publish.py --what models --execute.
# Prerequisite (local, before launching): python scripts/hf_publish.py --what cards
#                                         python scripts/hf_publish.py --what bundle --execute
# which puts the model cards, the copyright-gate inputs (scorecard, audit JSONs) and the sha256s of the provenance files
# approved by the local book-text scan at $BUNDLE_URI. HF_TOKEN comes from launch.py (box env); it is never printed.
# SCRIPT_ARGS = extra hf_publish.py args, e.g. "--models retain,undo_book_npofull".
#   python scripts/vast/launch.py up --job script --job-script scripts/vast/jobs/hf_upload.sh --tag hfup \
#       --min-gpu-ram 24 --max-dph 1.5
set -uo pipefail
source scripts/vast/common.sh
BUNDLE_URI=${BUNDLE_URI:-$BUCKET_URI/aya/hf_publish/bundle.tgz}
mkdir -p work/hf

stage "bundle (cards, gate inputs, approved file hashes)"
{ $PY -m lfm3u.storage get "$BUNDLE_URI" /tmp/hfbundle.tgz && tar xzf /tmp/hfbundle.tgz -C /root/lexsi; } || fail "bundle"
ls hf/model_cards

stage "huggingface_hub + hf_xet (torch / transformers untouched)"
$PY - <<'EOF' || $PY -m pip install -q --no-cache-dir --no-deps "huggingface_hub>=0.34" || fail "huggingface_hub"
import huggingface_hub as h
v = tuple(int(x) for x in h.__version__.split(".")[:2])
print("huggingface_hub", h.__version__)
raise SystemExit(0 if v >= (0, 34) else 1)
EOF
$PY -c "import hf_xet; print('hf_xet', hf_xet.__version__ if hasattr(hf_xet, '__version__') else 'ok')" 2>/dev/null || \
  $PY -m pip install -q --no-cache-dir --no-deps hf_xet || echo "hf_xet install failed (plain HTTP upload)"
$PY -c "import huggingface_hub as h; print('huggingface_hub', h.__version__)"
$PY -c "import dotenv" 2>/dev/null || $PY -m pip install -q --no-cache-dir python-dotenv
df -h /root/lexsi | tail -1

stage "upload models"
$PY scripts/hf_publish.py --what models --execute --workdir work/hf "$@" || fail "hf_publish models"
stage "hf upload done"
