#!/usr/bin/env bash
# Phase-6 audit runner. Each argument is a spec file (scripts/sweeps/audit_*.txt); each non-comment line is
#   NAME REF [scripts/10_audit.py args...]      REF = s3:// checkpoint prefix or HF id
# For every line: fetch the model, run 10_audit.py, mirror results/audit/NAME to the run prefix, delete the checkpoint.
# Needs the frozen data (DATA_URI), the attack sets (s3://<bucket>/attack_data) and the self-generated examples.
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/data/phase2-20261109.tgz}
SELFGEN_URI=${SELFGEN_URI:-$BUCKET_URI/unlearn/selfgen_v3.jsonl}
mkdir -p work models results/audit

stage "data + attack sets + selfgen"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
$PY -m lfm3u.storage download "$BUCKET_URI/attack_data" attack_data || fail "attack_data"
$PY -m lfm3u.storage get "$SELFGEN_URI" work/selfgen.jsonl || fail "selfgen"
$PY -c "import sympy, lm_eval; print('sympy', sympy.__version__)" || fail "deps"

failed() { echo "$1" >> results/audit/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }

for SPEC in "$@"; do
  [ -f "$SPEC" ] || fail "no spec file $SPEC"
  while read -r NAME REF EXTRA <&3; do
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    MP=$(fetch_model "$REF" "models/$NAME") || { failed "$NAME: download"; continue; }
    stage "audit $NAME ($REF) $EXTRA"
    eval "$PY scripts/10_audit.py --model \"$MP\" --name \"$NAME\" --out \"results/audit/$NAME\" $EXTRA" < /dev/null || failed "$NAME: audit"
    sync_up
    [[ $REF == s3://* ]] && rm -rf "models/$NAME"
  done 3< "$SPEC"
done
stage "audit done"
