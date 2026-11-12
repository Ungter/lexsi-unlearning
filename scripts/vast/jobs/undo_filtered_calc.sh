#!/usr/bin/env bash
# ERASURE_RESEARCH experiment 3 on one box: calculus-free UNDO distillation of the reference target (scripts/sweeps/u5.txt
# via unlearn_relearn.sh: distil, dev eval, upload, LoRA relearning), then the cross-subskill relearning curves (per-kind
# eval on the 243 test problems) on the uploaded student.
set -uo pipefail
source scripts/vast/common.sh
bash scripts/vast/jobs/unlearn_relearn.sh scripts/sweeps/u5.txt || fail "unlearn_relearn"
cat > /tmp/xs_filtered.txt <<EOF
12_relearn.py xs_undo_calc_filtered $BUCKET_URI/models/$RUN_ID/undo_calc_filtered --domain calc --kinds forget_diff,forget --calc-per-kind 81 --gen-bs 125
EOF
bash scripts/vast/jobs/attack_models.sh /tmp/xs_filtered.txt || fail "cross-subskill"
