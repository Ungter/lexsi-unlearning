#!/usr/bin/env bash
# Phase-4 sweep runner. Each argument is a sweep file (scripts/sweeps/*.txt); each non-comment line is one of:
#   selfgen [args]                 scripts/06_selfgen.py on the target -> uploaded to $SELFGEN_URI
#   eval NAME REF [args]           scripts/08_dev_eval.py on REF (s3:// prefix, HF id, or "target")
#   localize TARGET [args]         lfm3u.unlearn.localize on the target -> results/localize/TARGET, scores to $LOCALIZE_URI
#   NAME [07_unlearn.py args]      unlearn the target, dev-eval the result, upload the checkpoint, delete it locally
#                                  (--mask work/localize/T/scores.pt waits up to 90 min for $LOCALIZE_URI/T/scores.pt)
# A failing line is logged to results/unlearn/failed.txt and the sweep continues. Unlearn lines wait (up to 90 min)
# for $SELFGEN_URI when it does not exist yet, so sweep boxes can start together with the selfgen box.
set -uo pipefail
source scripts/vast/common.sh
TARGET_URI=${TARGET_URI:-$BUCKET_URI/models/script-20261109-044309/lfm3-muse-target-A}
SELFGEN_URI=${SELFGEN_URI:-$BUCKET_URI/unlearn/selfgen_v3.jsonl}
DATA_URI=${DATA_URI:-$BUCKET_URI/data/phase2-20261109.tgz}
LOCALIZE_URI=${LOCALIZE_URI:-$BUCKET_URI/localize/v1}
mkdir -p work models results/unlearn results/dev

stage "install safetune"
before=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
$PY -m pip install -q --no-cache-dir safetune || fail "pip install safetune"
after=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
[ "$before" = "$after" ] || fail "safetune changed torch/transformers ($before -> $after)"

stage "data + target"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
TARGET=$(fetch_model "$TARGET_URI" models/target) || fail "target download"

have_selfgen() {
  [ -f work/selfgen.jsonl ] && return 0
  $PY -m lfm3u.storage get "$SELFGEN_URI" work/selfgen.jsonl >/dev/null 2>&1
}
wait_selfgen() {
  local t=0
  until have_selfgen; do
    [ $t -ge 5400 ] && return 1
    [ $t -eq 0 ] && echo "waiting for $SELFGEN_URI"
    sleep 60; t=$((t + 60))
  done
}
wait_mask() {  # $1 = work/localize/T/scores.pt
  local t=0
  mkdir -p "$(dirname "$1")"
  until [ -f "$1" ] || $PY -m lfm3u.storage get "$LOCALIZE_URI/${1#work/localize/}" "$1" >/dev/null 2>&1; do
    [ $t -ge 5400 ] && return 1
    [ $t -eq 0 ] && echo "waiting for $LOCALIZE_URI/${1#work/localize/}"
    sleep 60; t=$((t + 60))
  done
}
failed() { echo "$1" >> results/unlearn/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }

for SWEEP in "$@"; do
  [ -f "$SWEEP" ] || fail "no sweep file $SWEEP"
  while read -r NAME REST <&3; do  # fd 3: commands in the loop must not consume the sweep file
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    case $NAME in
      selfgen)
        stage "selfgen $REST"
        $PY scripts/06_selfgen.py --model "$TARGET" --out work/selfgen.jsonl $REST || fail "selfgen"
        $PY -m lfm3u.storage put work/selfgen.jsonl "$SELFGEN_URI" || fail "selfgen upload"
        $PY -m lfm3u.storage put work/selfgen.jsonl.summary.json "$SELFGEN_URI.summary.json"
        cp work/selfgen.jsonl.summary.json results/unlearn/selfgen_summary.json
        sync_up ;;
      localize)
        read -r LT LARGS <<< "$REST"
        wait_selfgen || { failed "localize $LT: no selfgen"; continue; }
        stage "localize $LT $LARGS"
        $PY -m lfm3u.unlearn.localize --model "$TARGET" --target "$LT" --selfgen work/selfgen.jsonl \
          --out "results/localize/$LT" $LARGS < /dev/null || { failed "localize $LT"; continue; }
        mkdir -p "work/localize/$LT" && cp "results/localize/$LT/scores.pt" "work/localize/$LT/scores.pt"
        $PY -m lfm3u.storage put "work/localize/$LT/scores.pt" "$LOCALIZE_URI/$LT/scores.pt" || failed "localize $LT: upload"
        sync_up ;;
      eval)
        read -r ENAME REF EXTRA <<< "$REST"
        if [ "$REF" = target ]; then MP=$TARGET; else
          MP=$(fetch_model "$REF" "models/ref_$ENAME") || { failed "eval $ENAME: download"; continue; }
        fi
        stage "dev eval $ENAME ($REF) $EXTRA"
        eval "$PY scripts/08_dev_eval.py --model \"$MP\" --name \"$ENAME\" $EXTRA" < /dev/null || failed "eval $ENAME"
        sync_up ;;
      *)
        wait_selfgen || { failed "$NAME: no selfgen"; continue; }
        if [[ $REST =~ --mask[[:space:]]+([^[:space:]]+) ]]; then
          wait_mask "${BASH_REMATCH[1]}" || { failed "$NAME: no localizer scores"; continue; }
        fi
        stage "unlearn $NAME $REST"
        $PY scripts/07_unlearn.py --name "$NAME" --model "$TARGET" --source "$TARGET_URI" --selfgen work/selfgen.jsonl \
          --out "models/$NAME" --log "results/unlearn/$NAME/train.json" $REST < /dev/null || { failed "$NAME: unlearn"; continue; }
        stage "dev eval $NAME"
        $PY scripts/08_dev_eval.py --model "models/$NAME" --name "$NAME" < /dev/null || failed "$NAME: dev eval"
        stage "upload $NAME"
        upload_model "models/$NAME" "$NAME" || failed "$NAME: upload"
        rm -rf "models/$NAME"
        sync_up ;;
    esac
  done 3< "$SWEEP"
done
stage "sweep done"
