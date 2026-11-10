#!/usr/bin/env bash
# Unlearn -> dev eval -> upload -> relearning attack (attack 10), one box, one model at a time. Each argument is a sweep
# file; each non-comment line is one of:
#   NAME [07_unlearn.py args]              unlearn the reference target (scripts/07_unlearn.py)
#   undo NAME TEACHER_REF [15_undo args]   UNDO distillation of an unlearned checkpoint (scripts/15_undo.py)
# then scripts/08_dev_eval.py, upload to s3://<bucket>/models/$RUN_ID/NAME, and scripts/12_relearn.py on the result
# (domain from --target). Results: results/{unlearn,undo,dev,relearn}/NAME. A failing step is logged to
# results/failed.txt and the line moves on.
set -uo pipefail
source scripts/vast/common.sh
TARGET_URI=${TARGET_URI:-$BUCKET_URI/models/script-20261109-044309/lfm3-muse-target-A}
SELFGEN_URI=${SELFGEN_URI:-$BUCKET_URI/unlearn/selfgen_v3.jsonl}
SELFGEN_DEV_URI=${SELFGEN_DEV_URI:-$BUCKET_URI/unlearn/selfgen_dev_v3.jsonl}
DATA_URI=${DATA_URI:-$BUCKET_URI/data/phase2-20261109.tgz}
LOCALIZE_URI=${LOCALIZE_URI:-$BUCKET_URI/localize/v1}
FINEWEB_URI=${FINEWEB_URI:-$BUCKET_URI/eval/fineweb_sample.jsonl}
mkdir -p work models results/unlearn results/undo results/dev results/relearn

stage "install safetune + peft"
before=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
$PY -m pip install -q --no-cache-dir safetune peft || fail "pip install"
$PY -m pip install -q --no-cache-dir rouge-score >/dev/null 2>&1
after=$($PY -c "import torch, transformers; print(torch.__version__, transformers.__version__)")
[ "$before" = "$after" ] || fail "installs changed torch/transformers ($before -> $after)"

stage "data, target, selfgen, attack sets"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
TARGET=$(fetch_model "$TARGET_URI" models/target) || fail "target download"
$PY -m lfm3u.storage get "$SELFGEN_URI" work/selfgen.jsonl || fail "selfgen"
$PY -m lfm3u.storage get "$SELFGEN_DEV_URI" work/selfgen_dev.jsonl || fail "selfgen_dev"
$PY -m lfm3u.storage download "$BUCKET_URI/attack_data" attack_data >/dev/null || fail "attack_data"
if grep -qs '^undo ' "$@"; then
  stage "FineWeb shard for UNDO"
  $PY -m lfm3u.storage get "$FINEWEB_URI" work/fineweb_sample.jsonl || fail "fineweb held-out docs"
  $PY - <<'EOF' || fail "fineweb shard"
from huggingface_hub import HfApi, hf_hub_download
import os, shutil
files = sorted(f for f in HfApi().list_repo_files("HuggingFaceFW/fineweb", repo_type="dataset") if f.startswith("sample/10BT/"))
p = hf_hub_download("HuggingFaceFW/fineweb", files[-1], repo_type="dataset", local_dir="work/fw")
shutil.move(p, "work/fineweb_shard.parquet")
print("fineweb shard", files[-1], os.path.getsize("work/fineweb_shard.parquet") // 2**20, "MB")
EOF
fi

failed() { echo "$1" >> results/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }
wait_mask() {
  local t=0
  mkdir -p "$(dirname "$1")"
  until [ -f "$1" ] || $PY -m lfm3u.storage get "$LOCALIZE_URI/${1#work/localize/}" "$1" >/dev/null 2>&1; do
    [ $t -ge 1800 ] && return 1
    sleep 60; t=$((t + 60))
  done
}
evaluate() {  # NAME TARGET(book|calculus)
  local name=$1 dom=book
  [ "$2" = calculus ] && dom=calc
  stage "dev eval $name"
  $PY scripts/08_dev_eval.py --model "models/$name" --name "$name" < /dev/null || failed "$name: dev eval"
  stage "upload $name"
  upload_model "models/$name" "$name" || failed "$name: upload"
  sync_up
  stage "relearn $name ($dom)"
  $PY scripts/12_relearn.py --model "models/$name" --name "$name" --out "results/relearn/$name" --domain "$dom" < /dev/null \
    || failed "$name: relearn"
  rm -rf "models/$name"
  sync_up
}

for SWEEP in "$@"; do
  [ -f "$SWEEP" ] || fail "no sweep file $SWEEP"
  while read -r NAME REST <&3; do
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    if [ "$NAME" = undo ]; then
      read -r UNAME TREF UARGS <<< "$REST"
      [[ $UARGS =~ --target[[:space:]]+([a-z]+) ]] && UT=${BASH_REMATCH[1]} || { failed "undo $UNAME: no --target"; continue; }
      TP=$(fetch_model "$TREF" "models/teacher_$UNAME") || { failed "undo $UNAME: teacher download"; continue; }
      stage "undo $UNAME ($TREF) $UARGS"
      $PY scripts/15_undo.py --teacher "$TP" --source "$TREF" --name "$UNAME" --out "models/$UNAME" \
        --fineweb work/fineweb_shard.parquet --log "results/undo/$UNAME/train.json" $UARGS < /dev/null \
        || { failed "undo $UNAME"; sync_up; continue; }
      rm -rf "models/teacher_$UNAME"
      evaluate "$UNAME" "$UT"
      continue
    fi
    if [[ $REST =~ --mask[[:space:]]+([^[:space:]]+) ]]; then
      wait_mask "${BASH_REMATCH[1]}" || { failed "$NAME: no localizer scores"; continue; }
    fi
    [[ $REST =~ --target[[:space:]]+([a-z]+) ]] && T=${BASH_REMATCH[1]} || { failed "$NAME: no --target"; continue; }
    stage "unlearn $NAME $REST"
    $PY scripts/07_unlearn.py --name "$NAME" --model "$TARGET" --source "$TARGET_URI" --selfgen work/selfgen.jsonl \
      --out "models/$NAME" --log "results/unlearn/$NAME/train.json" $REST < /dev/null || { failed "$NAME: unlearn"; continue; }
    evaluate "$NAME" "$T"
  done 3< "$SWEEP"
done
stage "unlearn+relearn done"
