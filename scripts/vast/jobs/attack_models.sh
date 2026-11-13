#!/usr/bin/env bash
# Phase-6 HF-based attacks (8 abliteration, 10 relearning). Each argument is a spec file; each non-comment line is
#   SCRIPT NAME REF [script args...]      SCRIPT = 11_abliterate.py | 12_relearn.py | 17_local_abl.py ; REF = s3:// prefix or HF id
# Results: results/{abl,relearn,local_abl}/NAME (mirrored to the run prefix). The relearning script needs the dev-split
# self-generated solutions (selfgen_dev_v3.jsonl, written by selfgen_dev.sh); lines wait up to 30 min for it.
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/data/phase2-20261109.tgz}
SELFGEN_DEV_URI=${SELFGEN_DEV_URI:-$BUCKET_URI/unlearn/selfgen_dev_v3.jsonl}
mkdir -p work models results

stage "install peft"
before=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
$PY -m pip install -q --no-cache-dir peft || fail "pip install peft"
after=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
[ "$before" = "$after" ] || fail "peft changed torch/transformers ($before -> $after)"
$PY -m pip install -q --no-cache-dir rouge-score >/dev/null 2>&1

stage "data + attack sets"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
$PY -m lfm3u.storage download "$BUCKET_URI/attack_data" attack_data || fail "attack_data"

wait_selfgen_dev() {
  local t=0
  until [ -f work/selfgen_dev.jsonl ] || $PY -m lfm3u.storage get "$SELFGEN_DEV_URI" work/selfgen_dev.jsonl >/dev/null 2>&1; do
    [ $t -ge 1800 ] && return 1
    [ $t -eq 0 ] && echo "waiting for $SELFGEN_DEV_URI"
    sleep 30; t=$((t + 30))
  done
}
failed() { echo "$1" >> results/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }

for SPEC in "$@"; do
  [ -f "$SPEC" ] || fail "no spec file $SPEC"
  while read -r SCRIPT NAME REF EXTRA <&3; do
    { [ -z "$SCRIPT" ] || [[ $SCRIPT == \#* ]]; } && continue
    KIND=${SCRIPT%.py}; KIND=${KIND#*_}
    case $KIND in abliterate) KIND=abl ;; esac
    if { [ "$KIND" = relearn ] || [ "$KIND" = local_abl ]; } && [[ $EXTRA == *calc* ]]; then wait_selfgen_dev || { failed "$NAME: no selfgen_dev"; continue; }; fi
    DIR="models/$(echo "$REF" | md5sum | cut -c1-12)"  # one download per checkpoint, shared by the lines of a spec
    MP=$(fetch_model "$REF" "$DIR") || { failed "$NAME: download"; continue; }
    stage "$SCRIPT $NAME ($REF) $EXTRA"
    eval "$PY scripts/$SCRIPT --model \"$MP\" --name \"$NAME\" --out \"results/$KIND/$NAME\" $EXTRA" < /dev/null || failed "$SCRIPT $NAME"
    sync_up
  done 3< "$SPEC"
  rm -rf models/*
done
stage "attacks done"
