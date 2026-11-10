#!/usr/bin/env bash
# Phase-6 attack 13 (J-lens readout) on unlearned checkpoints. Spec lines: NAME REF
# For each: (a) the BASE model's lens applied to the checkpoint's activations, (b) the checkpoint's OWN lens (fit on 100
# FineWeb prompts). Both run the multi-hop/calculus/Harry Potter validation sets and the knowmem hidden readout
# (questions: knowmem_forget_qa_jspace_v3.jsonl). Outputs results/jspace/<NAME>_baselens and <NAME>_ownlens.
set -uo pipefail
source scripts/vast/common.sh
JLENS_SHA=581d398613e5602a5af361e1c34d3a92ea82ba8e
BASE_LENS_URI=${BASE_LENS_URI:-$BUCKET_URI/jspace/script-20261109-034952/base/lens.pt}
QUESTIONS_URI=${QUESTIONS_URI:-$BUCKET_URI/jspace/inputs/knowmem_forget_qa_jspace_v3.jsonl}
mkdir -p work/jspace models results/jspace

stage "install jlens@${JLENS_SHA:0:7} (--no-deps)"
versions() { $PY -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"; }
BEFORE=$(versions)
$PY -m pip install -q --no-cache-dir --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" || fail "jlens install"
$PY -c "import matplotlib" 2>/dev/null || $PY -m pip install -q --no-cache-dir matplotlib
[ "$BEFORE" = "$(versions)" ] || fail "an install changed torch/transformers/numpy"

stage "inputs"
$PY -m lfm3u.storage get "$BUCKET_URI/eval/fineweb_sample.jsonl" work/fineweb.jsonl || fail "fineweb"
$PY -m lfm3u.storage get "$QUESTIONS_URI" work/knowmem.jsonl || fail "questions"
$PY -m lfm3u.storage get "$BASE_LENS_URI" work/base_lens.pt || fail "base lens"

failed() { echo "$1" >> results/jspace/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }
COMMON=(--corpus work/fineweb.jsonl --questions work/knowmem.jsonl --out results/jspace --min-prompts 100 --max-prompts 1000)

for SPEC in "$@"; do
  while read -r NAME REF <&3; do
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    MP=$(fetch_model "$REF" "models/$NAME") || { failed "$NAME: download"; continue; }
    stage "$NAME: base lens on the checkpoint"
    $PY scripts/05_jspace.py --name "${NAME}_baselens" --model "$MP" --source "$REF" --lens work/base_lens.pt \
      --lens-dir work/jspace "${COMMON[@]}" --stages validate,hidden < /dev/null || failed "$NAME: baselens"
    sync_up
    stage "$NAME: own lens (fit + readouts)"
    $PY scripts/05_jspace.py --name "${NAME}_ownlens" --model "$MP" --source "$REF" --lens-dir work/jspace \
      "${COMMON[@]}" --fit-seconds 1500 --stages fit,validate,hidden < /dev/null || failed "$NAME: ownlens"
    rm -f "work/jspace/${NAME}_ownlens/lens.pt"
    sync_up
    rm -rf "models/$NAME"
  done 3< "$SPEC"
done
stage "jspace attack done"
