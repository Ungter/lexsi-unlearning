#!/usr/bin/env bash
# ERASURE_RESEARCH experiment 3 on one box: calculus-free UNDO distillation of the reference target (scripts/sweeps/u5.txt
# via unlearn_relearn.sh: distil, dev eval, upload, LoRA relearning), then the cross-subskill relearning curves (per-kind
# eval on the 243 test problems) on the uploaded student.
# Tiny Aya: TARGET_URI must be set (see unlearn_relearn.sh); --gen-bs 64 (LFM3 125): the KV cache of 36 attention layers
# (4 KV heads x 128) is ~74 KB per token, ~3.4k tokens per sequence with the default system preamble.
set -uo pipefail
source scripts/vast/common.sh
bash scripts/vast/jobs/unlearn_relearn.sh scripts/sweeps/u5.txt || fail "unlearn_relearn"
cat > /tmp/xs_filtered.txt <<EOF
12_relearn.py xs_undo_calc_filtered $BUCKET_URI/models/$RUN_ID/undo_calc_filtered --domain calc --kinds forget_diff,forget --calc-per-kind 81 --gen-bs 64
EOF
bash scripts/vast/jobs/attack_models.sh /tmp/xs_filtered.txt || fail "cross-subskill"
