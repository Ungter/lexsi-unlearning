#!/usr/bin/env bash
# AuditKIT (Lexsi Labs, pip auditkit) evaluation runner. Each argument is a spec file (scripts/sweeps/aya_auditkit*.txt);
# each non-comment line is
#   NAME REF [scripts/22_auditkit_eval.py args...]     REF = s3:// checkpoint prefix or HF id
# For every line: fetch the model, run scripts/22_auditkit_eval.py, mirror results/auditkit/NAME to the run prefix,
# delete the checkpoint. Env (launch.py --extra-env):
#   ONLY=a+b+c      run only these spec names (split one spec over several boxes)
#   AK_ARGS=--smoke extra args for every 22_auditkit_eval.py call ('+' = space)
# Installs auditkit[lmeval]==1.1.12 with torch/transformers/vllm/numpy/tokenizers pinned to the image's versions
# (constraints file); if pip cannot resolve that, falls back to --no-deps installs. Fails if any pinned version changed.
set -uo pipefail
source scripts/vast/common.sh
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
AK_VERSION=${AK_VERSION:-1.1.12}
AK_ARGS=${AK_ARGS:-}
AK_ARGS=${AK_ARGS//+/ }
ONLY=${ONLY:-}
mkdir -p work models results/auditkit

stage "data"
if [ ! -f data/MANIFEST.json ]; then
  { $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
fi

stage "install auditkit[lmeval]==$AK_VERSION"
versions() { $PY -c "import torch, transformers, numpy, tokenizers, vllm; print(torch.__version__, transformers.__version__, numpy.__version__, tokenizers.__version__, vllm.__version__)"; }
BEFORE=$(versions)
$PY -m pip freeze 2>/dev/null | grep -iE '^(torch|transformers|vllm|numpy|tokenizers)==' > /tmp/ak_pins.txt
cat /tmp/ak_pins.txt
if ! $PY -m pip install -q --no-cache-dir -c /tmp/ak_pins.txt "auditkit[lmeval]==$AK_VERSION" "lm-eval>=0.4.9"; then
  echo "constrained install failed; falling back to --no-deps"
  $PY -m pip install -q --no-cache-dir --no-deps "auditkit==$AK_VERSION" lm-eval evaluate sqlitedict pytablewriter sacrebleu \
    word2number more_itertools dill accelerate DataProperty mbstrdecoder pathvalidate tabledata tcolorpy typepy chardet \
    portalocker colorama tabulate || fail "auditkit install"
fi
AFTER=$(versions)
echo "torch/transformers/numpy/tokenizers/vllm: $BEFORE -> $AFTER"
[ "$BEFORE" = "$AFTER" ] || fail "auditkit install changed torch/transformers/numpy/tokenizers/vllm"
$PY -c "import auditkit, lm_eval, accelerate, sympy, rouge_score, evaluate, sqlitedict, pytablewriter, sacrebleu, word2number, more_itertools, datasets; print('auditkit', auditkit.__version__, 'lm_eval', getattr(lm_eval, '__version__', '?'))" || fail "deps"
$PY -m auditkit.compat 2>&1 | tail -n 15 || true   # AuditKIT's own environment check (informational)

failed() { echo "$1" >> results/auditkit/failed.txt; echo "== $(stamp) FAILED (continuing): $1"; sync_up; }

for SPEC in "$@"; do
  [ -f "$SPEC" ] || fail "no spec file $SPEC"
  while read -r NAME REF EXTRA <&3; do
    { [ -z "$NAME" ] || [[ $NAME == \#* ]]; } && continue
    if [ -n "$ONLY" ] && [[ "+$ONLY+" != *"+$NAME+"* ]]; then continue; fi
    MP=$(fetch_model "$REF" "models/$NAME") || { failed "$NAME: download"; continue; }
    stage "auditkit $NAME ($REF) $EXTRA $AK_ARGS"
    t0=$(date +%s)
    eval "$PY scripts/22_auditkit_eval.py --model \"$MP\" --name \"$NAME\" --out \"results/auditkit/$NAME\" $EXTRA $AK_ARGS" < /dev/null \
      || failed "$NAME: 22_auditkit_eval rc=$? (see results/auditkit/$NAME/summary.json)"
    echo "== $NAME took $(( $(date +%s) - t0 ))s"
    sync_up
    [[ $REF == s3://* ]] && rm -rf "models/$NAME"
  done 3< "$SPEC"
done
stage "auditkit done"
