#!/usr/bin/env bash
# Verbatim book recall under prompt attacks (scripts/23_verbatim_attacks.py; review issue M7).
# Each argument is a spec file (scripts/sweeps/aya_verbatk_*.txt); each non-comment line is
#   NAME REF [scripts/23_verbatim_attacks.py args...]      REF = s3:// checkpoint prefix or HF id
# For every line: fetch the model, run 23_verbatim_attacks.py -> results/verbatk/NAME, mirror results to the run
# prefix, delete the checkpoint. Needs the frozen aya data (book/verbmem_forget_{test,train}.jsonl) and
# $BUCKET_URI/aya/attack_data/verbatim_paraphrase/test.jsonl (hand-written paraphrases / French tails; never published).
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
PARA_URI=${PARA_URI:-$BUCKET_URI/aya/attack_data/verbatim_paraphrase}
mkdir -p work models results/verbatk

stage "data + verbatim paraphrases"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
$PY -m lfm3u.storage download "$PARA_URI" attack_data/verbatim_paraphrase || fail "attack_data/verbatim_paraphrase"
[ -f attack_data/verbatim_paraphrase/test.jsonl ] || fail "no verbatim_paraphrase/test.jsonl"

failed() { echo "$1" >> results/verbatk/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }

for SPEC in "$@"; do
  [ -f "$SPEC" ] || fail "no spec file $SPEC"
  while read -r NAME REF EXTRA <&3; do
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    MP=$(fetch_model "$REF" "models/$NAME") || { failed "$NAME: download"; continue; }
    stage "verbatk $NAME ($REF) $EXTRA"
    eval "$PY scripts/23_verbatim_attacks.py --model \"$MP\" --name \"$NAME\" --out \"results/verbatk/$NAME\" $EXTRA" < /dev/null \
      || failed "$NAME: verbatk"
    sync_up
    [[ $REF == s3://* ]] && rm -rf "models/$NAME"
  done 3< "$SPEC"
done
stage "verbatk done"
