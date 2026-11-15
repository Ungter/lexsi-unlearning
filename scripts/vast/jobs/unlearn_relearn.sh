#!/usr/bin/env bash
# Unlearn -> dev eval -> upload -> relearning attack (attack 10), one box, one model at a time. Each argument is a sweep
# file; each non-comment line is one of:
#   NAME [07_unlearn.py args]              unlearn the reference target (scripts/07_unlearn.py)
#   undo NAME TEACHER_REF [15_undo args]   UNDO distillation of an unlearned checkpoint (scripts/15_undo.py);
#                                          TEACHER_REF = s3 uri (shell variables such as $BUCKET_URI / $TEACHER_RUN are
#                                          expanded) or "target" = the reference target ($TARGET_URI, already local)
# then scripts/08_dev_eval.py, upload to s3://<bucket>/models/$RUN_ID/NAME, and scripts/12_relearn.py on the result
# (domain from --target). Results: results/{unlearn,undo,dev,relearn}/NAME. A failing step is logged to
# results/failed.txt and the line moves on.
# Tiny Aya (aya branch): the LFM3 defaults (lfm3-muse-target-A, selfgen*_v3, localize/v1, phase2-20261109.tgz) hold LFM3
# checkpoints / LFM3 token ids and must not be reused. TARGET_URI is required (the Aya MUSE target from 03_finetune);
# the other artefacts default to the aya/ prefix of the bucket. FineWeb docs are plain text and stay shared.
set -uo pipefail
source scripts/vast/common.sh
TARGET_URI=${TARGET_URI:?set TARGET_URI to the Tiny Aya MUSE target checkpoint (s3://.../aya-muse-target)}
SELFGEN_URI=${SELFGEN_URI:-$BUCKET_URI/aya/unlearn/selfgen.jsonl}
SELFGEN_DEV_URI=${SELFGEN_DEV_URI:-$BUCKET_URI/aya/unlearn/selfgen_dev.jsonl}
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
ATTACK_URI=${ATTACK_URI:-$BUCKET_URI/aya/attack_data}  # ALGEBRA attack sets (arch.MATH_TARGET), not the LFM3 calculus ones
LOCALIZE_URI=${LOCALIZE_URI:-$BUCKET_URI/aya/localize/v1}
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
$PY -m lfm3u.storage download "$ATTACK_URI" attack_data >/dev/null || fail "attack_data"
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
evaluate() {  # NAME TARGET(book|calculus|algebra|arithmetic|...)
  local name=$1 dom=calc
  [ "$2" = book ] && dom=book  # every math target (algebra = arch.MATH_TARGET here) is the 'calc' domain
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
      if [ "$TREF" = target ]; then
        TREF=$TARGET_URI; TP=$TARGET
      else
        TREF=$(eval echo "$TREF") || { failed "undo $UNAME: teacher ref (unset variable?)"; continue; }
        TP=$(fetch_model "$TREF" "models/teacher_$UNAME") || { failed "undo $UNAME: teacher download"; continue; }
      fi
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
