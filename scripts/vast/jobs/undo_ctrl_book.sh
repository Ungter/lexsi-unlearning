#!/usr/bin/env bash
# DECISIONS TODO 3 on one box: UNDO distillation of the reference target on the book mixture (scripts/sweeps/u4.txt via
# unlearn_relearn.sh: distil, dev eval, upload, LoRA relearning), then the strong book attack used for every other book
# model (all weights, lr 1e-5, + 200 forget-text chunks; as rlfx_*) on the uploaded student.
set -uo pipefail
source scripts/vast/common.sh
bash scripts/vast/jobs/unlearn_relearn.sh scripts/sweeps/u4.txt || fail "unlearn_relearn"
cat > /tmp/rlfx_ctrl.txt <<EOF
12_relearn.py rlfx_undo_book_ctrl $BUCKET_URI/models/$RUN_ID/undo_book_ctrl --domain book --kinds forget --full --lr 1e-5 --book-extra 200 --steps 0,25,50,100,200
EOF
bash scripts/vast/jobs/attack_models.sh /tmp/rlfx_ctrl.txt || fail "strong attack"
