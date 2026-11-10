#!/usr/bin/env bash
# PLAN attack 14 (J-space concept injection) on the calculus target. Spec lines: NAME REF LENS STAGES
#   LENS   target | base         (target lens for the target and every unlearned checkpoint; base lens for base)
#   STAGES sweep,eval | eval     (the first line must be the target with sweep,eval: it writes config.json, which
#                                 every later eval-only line reuses; the config is chosen on the target's DEV split only)
# Outputs results/jspace/inject_calc/<NAME>/ (sweep.json, config.json, summary.json, generations/*.jsonl).
set -uo pipefail
source scripts/vast/common.sh
JLENS_SHA=581d398613e5602a5af361e1c34d3a92ea82ba8e
BASE_LENS_URI=${BASE_LENS_URI:-$BUCKET_URI/jspace/script-20261109-034952/base/lens.pt}
TARGET_LENS_URI=${TARGET_LENS_URI:-$BUCKET_URI/jspace/script-20261109-065711/target_v3/lens.pt}
OUT=results/jspace/inject_calc
CONFIG=$OUT/target/config.json
mkdir -p work models attack_data/_inputs "$OUT"

stage "install jlens@${JLENS_SHA:0:7} (--no-deps)"
versions() { $PY -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"; }
BEFORE=$(versions)
$PY -m pip install -q --no-cache-dir --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" || fail "jlens install"
[ "$BEFORE" = "$(versions)" ] || fail "an install changed torch/transformers/numpy"
$PY -c "import sympy; print('sympy', sympy.__version__)" || fail "deps"

stage "inputs"
for F in math_calc_dev.jsonl math_calc_test.jsonl; do
  $PY -m lfm3u.storage get "$BUCKET_URI/attack_data/_inputs/$F" "attack_data/_inputs/$F" || fail "attack data $F"
done
$PY -m lfm3u.storage get "$TARGET_LENS_URI" work/target_lens.pt || fail "target lens"
$PY -m lfm3u.storage get "$BASE_LENS_URI" work/base_lens.pt || fail "base lens"

failed() { echo "$1" >> "$OUT/failed.txt"; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }
SMOKED=0

for SPEC in "$@"; do
  [ -f "$SPEC" ] || fail "no spec file $SPEC"
  while read -r NAME REF LENS STAGES <&3; do
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    MP=$(fetch_model "$REF" "models/$NAME") || { failed "$NAME: download"; continue; }
    if [ "$SMOKED" = 0 ]; then
      stage "smoke: $NAME, 1 problem per kind, 32 tokens"
      $PY scripts/14_jspace_inject.py --model "$MP" --name smoke --lens "work/${LENS}_lens.pt" --out work/smoke \
        --stages sweep,eval --layers 8,5-6 --alphas 0.5 --dev-per-kind 1 --n-test-per-kind 1 --max-tokens 32 \
        --sweep-max-tokens 32 < /dev/null || fail "smoke"
      SMOKED=1
    fi
    ARGS=(--model "$MP" --name "$NAME" --lens "work/${LENS}_lens.pt" --out "$OUT/$NAME" --stages "$STAGES" --batch 192)
    if [[ $STAGES != *sweep* ]]; then
      [ -f "$CONFIG" ] || { failed "$NAME: no $CONFIG (target sweep missing)"; continue; }
      ARGS+=(--config "$CONFIG")
    fi
    stage "inject $NAME ($REF) lens=$LENS stages=$STAGES"
    $PY scripts/14_jspace_inject.py "${ARGS[@]}" < /dev/null || failed "$NAME: inject"
    sync_up
    [[ $REF == s3://* ]] && rm -rf "models/$NAME"
  done 3< "$SPEC"
done
stage "jspace inject done"
