#!/usr/bin/env bash
# Phase-6 audit runner. Each argument is a spec file (scripts/sweeps/audit_*.txt); each non-comment line is
#   NAME REF [scripts/10_audit.py args...]      REF = s3:// checkpoint prefix or HF id
# For every line: fetch the model, run 10_audit.py, mirror results/audit/NAME to the run prefix, delete the checkpoint.
# Needs the frozen data (DATA_URI), the attack sets (ATTACK_URI) and the self-generated examples.
# Tiny Aya (aya branch): data / selfgen default to the aya/ prefix of the bucket (the LFM3 phase2-20261109.tgz and
# selfgen_v3 hold LFM3 token ids / LFM3 generations). attack_data is the ALGEBRA build (arch.MATH_TARGET) under
# $BUCKET_URI/aya/attack_data, NOT the LFM3 calculus sets in $BUCKET_URI/attack_data (10_audit.py rejects those).
# 10_audit.py clamps --max-model-len to 8192 and maps the LFM3 *_think tasks to their direct variants.
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
SELFGEN_URI=${SELFGEN_URI:-$BUCKET_URI/aya/unlearn/selfgen.jsonl}
ATTACK_URI=${ATTACK_URI:-$BUCKET_URI/aya/attack_data}
mkdir -p work models results/audit

stage "data + attack sets + selfgen"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi
$PY -m lfm3u.storage download "$ATTACK_URI" attack_data || fail "attack_data"
t=0  # wait up to 90 min for the selfgen file, so audit boxes can start together with the prep box
until $PY -m lfm3u.storage get "$SELFGEN_URI" work/selfgen.jsonl >/dev/null 2>&1; do
  [ $t -ge 5400 ] && fail "selfgen"
  [ $t -eq 0 ] && echo "waiting for $SELFGEN_URI"
  sleep 60; t=$((t + 60))
done
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
