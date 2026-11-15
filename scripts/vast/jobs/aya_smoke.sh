#!/usr/bin/env bash
# Tiny Aya GPU smoke test: every pipeline step after the fine-tune, with tiny budgets, on the BASE model as a stand-in
# target. A failing step logs "SMOKE FAIL <step>" and the job continues; results are synced after every step.
#   python scripts/vast/launch.py up --job script --job-script scripts/vast/jobs/aya_smoke.sh --min-gpu-ram 90 \
#       --extra-env STEPS=selfgen+localize+unlearn+deveval
# STEPS (+ or comma list, default all; launch.py --extra-env splits on commas, so use +): selfgen selfgen_dev localize unlearn rmu deveval relearn audit abl local_abl weight
#   undo selftest jspace   (rmu = only the rmu unlearn, which abl/local_abl/weight/relearn attack)
# Artefacts: s3://<bucket>/aya/smoke/ (selfgen, localize scores).
set -uo pipefail
source scripts/vast/common.sh
BASE_ID=${TARGET_URI:-CohereLabs/tiny-aya-global}
DATA_URI=${DATA_URI:-$BUCKET_URI/aya/data/phase2.tgz}
ATTACK_URI=${ATTACK_URI:-$BUCKET_URI/aya/attack_data}  # ALGEBRA attack sets (arch.MATH_TARGET), not the LFM3 calculus ones
SMOKE_URI=${SMOKE_URI:-$BUCKET_URI/aya/smoke}
STEPS=${STEPS:-all}
MT=${MATH_TARGET:-algebra}  # math unlearning target (aya: algebra; calculus also works for 07_unlearn)
mkdir -p work models results/smoke
FAILS=""
want() { local s=,${STEPS//+/,},; [[ $s == *,all,* || $s == *,$1,* ]]; }
run() {  # run NAME CMD...: timed, continue on failure
  local name=$1; shift
  stage "SMOKE START $name"
  local t0=$(date +%s)
  if "$@" < /dev/null; then
    echo "SMOKE PASS $name $(( $(date +%s) - t0 ))s"
  else
    echo "SMOKE FAIL $name $(( $(date +%s) - t0 ))s"
    FAILS="$FAILS $name"
  fi
  echo "$name $(( $(date +%s) - t0 ))" >> results/smoke/timings.txt
  sync_up
}

stage "setup: installs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
versions() { $PY -c "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"; }
BEFORE=$(versions)
$PY -m pip install -q --no-cache-dir safetune peft || echo "SMOKE FAIL pip safetune peft"
$PY -m pip install -q --no-cache-dir rouge-score pyarrow >/dev/null 2>&1
if want jspace || want selftest; then
  JLENS_SHA=581d398613e5602a5af361e1c34d3a92ea82ba8e
  $PY -m pip install -q --no-cache-dir --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" \
    || uv pip install --system -q --no-deps "https://github.com/anthropics/jacobian-lens/archive/$JLENS_SHA.tar.gz" \
    || echo "SMOKE FAIL jlens install"
  $PY -c "import matplotlib" 2>/dev/null || $PY -m pip install -q --no-cache-dir matplotlib >/dev/null 2>&1
fi
AFTER=$(versions)
echo "torch/transformers/numpy: $BEFORE -> $AFTER"
[ "$BEFORE" = "$AFTER" ] || echo "SMOKE WARN installs changed torch/transformers/numpy"

stage "setup: data, base model, attack sets"
{ $PY -m lfm3u.storage get "$DATA_URI" /tmp/data.tgz && tar xzf /tmp/data.tgz -C /root/lexsi; } || fail "data"
$PY -c "from huggingface_hub import snapshot_download; snapshot_download('$BASE_ID', local_dir='models/base')" || fail "base download"
ln -sfn base models/target  # lfm3u.surgery default TARGET_MODEL
T=models/base
$PY -m lfm3u.storage download "$ATTACK_URI" attack_data >/dev/null || echo "SMOKE FAIL attack_data"
$PY -m lfm3u.storage get "$BUCKET_URI/eval/fineweb_sample.jsonl" work/fineweb_sample.jsonl || echo "SMOKE FAIL fineweb sample"
$PY -m lfm3u.storage get "$BUCKET_URI/eval/fineweb_anchor.jsonl" work/fineweb_anchor.jsonl || echo "SMOKE FAIL fineweb anchor"
sync_up

# ------------------------------------------------------------------ selfgen
if want selfgen; then
  run selfgen $PY scripts/06_selfgen.py --model $T --out work/selfgen.jsonl --limit 6
  [ -f work/selfgen.jsonl ] && $PY -m lfm3u.storage put work/selfgen.jsonl "$SMOKE_URI/selfgen.jsonl"
  cp work/selfgen.jsonl.summary.json results/smoke/selfgen_summary.json 2>/dev/null
fi
if want selfgen_dev || want relearn || want local_abl || want selftest; then
  run selfgen_dev $PY scripts/06_selfgen.py --model $T --split dev --out work/selfgen_dev.jsonl --limit 6
  [ -f work/selfgen_dev.jsonl ] && $PY -m lfm3u.storage put work/selfgen_dev.jsonl "$SMOKE_URI/selfgen_dev.jsonl"
fi
if [ ! -s work/selfgen.jsonl ]; then  # another box may have produced it
  $PY -m lfm3u.storage get "$SMOKE_URI/selfgen.jsonl" work/selfgen.jsonl >/dev/null 2>&1
fi

# ------------------------------------------------------------------ localize
if want localize; then
  for LT in book $MT; do
    run "localize_$LT" $PY -m lfm3u.unlearn.localize --model $T --target $LT --selfgen work/selfgen.jsonl \
      --out results/localize/$LT --n-forget 8 --n-retain 8 --max-len 1024
    if [ -f results/localize/$LT/scores.pt ]; then
      mkdir -p work/localize/$LT && cp results/localize/$LT/scores.pt work/localize/$LT/scores.pt
      $PY -m lfm3u.storage put work/localize/$LT/scores.pt "$SMOKE_URI/localize/$LT/scores.pt"
      rm -f results/localize/$LT/scores.pt  # keep results/ small
    fi
  done
fi

# ------------------------------------------------------------------ unlearn
unlearn() {  # NAME [07 args]
  local name=$1; shift
  run "unlearn_$name" $PY scripts/07_unlearn.py --name "sm_$name" --model $T --source "$BASE_ID" --selfgen work/selfgen.jsonl \
    --out "models/sm_$name" --log "results/unlearn/sm_$name/train.json" "$@"
  [ -f "results/unlearn/sm_$name/train.json" ] && grep -E '"(peak_gpu_gb|train_seconds|finite)"' "results/unlearn/sm_$name/train.json"
}
if want unlearn; then
  unlearn npo --target book --method npo --lr 5e-5 --epochs 1 --n-forget 8 --n-retain 8
  rm -rf models/sm_npo
  unlearn npo_full --target $MT --method npo_full --lr 1e-5 --epochs 1 --steps 3 --n-forget 8 --n-retain 8
  unlearn npo_sam --target $MT --method npo_sam --lr 1e-5 --epochs 1 --steps 3 --n-forget 8 --n-retain 8 --no-save
  unlearn ga --target book --method ga --lr 1e-5 --epochs 1 --steps 3 --n-forget 8 --n-retain 8 --no-save
  unlearn graddiff --target $MT --method graddiff --lr 1e-5 --epochs 1 --steps 3 --n-forget 8 --n-retain 8 --no-save
  unlearn gakl --target book --method gakl --lr 1e-5 --epochs 1 --steps 3 --n-forget 8 --n-retain 8 --no-save
  if [ -f work/localize/$MT/scores.pt ]; then
    unlearn npo_full_mask --target $MT --method npo_full --lr 1e-5 --epochs 1 --steps 3 --n-forget 8 --n-retain 8 \
      --mask work/localize/$MT/scores.pt --no-save
    unlearn rmu_mask --target $MT --method rmu --layer 6 --steps 3 --n-forget 8 --n-retain 8 \
      --mask work/localize/$MT/scores.pt --no-save
  else
    echo "SMOKE FAIL unlearn_npo_full_mask (no localize scores)"
  fi
  if [ -f work/localize/book/scores.pt ]; then
    unlearn ablate_book --target book --method ablate --mask work/localize/book/scores.pt --n-forget 8 --n-retain 8 --no-save
  fi
fi
if want unlearn || want rmu; then
  unlearn rmu --target $MT --method rmu --layer 6 --lr 5e-5 --steps 3 --n-forget 8 --n-retain 8
fi

# ------------------------------------------------------------------ dev eval
if want deveval; then
  M=models/sm_npo_full; [ -f $M/config.json ] || M=models/sm_rmu; [ -f $M/config.json ] || M=$T
  run deveval $PY scripts/08_dev_eval.py --model $M --name sm_deveval --limit 5
fi
rm -rf models/sm_npo_full

# ------------------------------------------------------------------ attacks (on the rmu checkpoint, else base)
A=models/sm_rmu; [ -f $A/config.json ] || A=$T
if want relearn; then
  run relearn_calc $PY scripts/12_relearn.py --model $A --name sm_rl_calc --out results/relearn/sm_rl_calc --domain calc \
    --steps 0,2 --n-calc 4 --max-new 256 --forget-n 4 --related-n 4 --gen-bs 8
  run relearn_book $PY scripts/12_relearn.py --model $A --name sm_rl_book --out results/relearn/sm_rl_book --domain book \
    --kinds forget --steps 0,2
  run relearn_full $PY scripts/12_relearn.py --model $A --name sm_rl_full --out results/relearn/sm_rl_full --domain calc \
    --kinds forget --steps 0,2 --n-calc 4 --max-new 256 --forget-n 4 --gen-bs 8 --full
fi
if want audit; then
  run audit $PY scripts/10_audit.py --model $A --name sm_audit --limit 2 --direct-max 512 --chat-max 64 --ifeval-max 256 --bon-n 2
  $PY - <<'EOF'
import json
r = json.load(open("results/audit/sm_audit/audit.json"))
bad = {k: v.get("error") for k, v in r["results"].items() if isinstance(v, dict) and v.get("error")}
print("AUDIT task errors:", json.dumps(bad, indent=1) if bad else "none")
for k, v in r["results"].items():
    if isinstance(v, dict) and v.get("error"):
        print("----", k, "\n", v.get("traceback"))
EOF
fi
if want abl; then
  run abl_calc $PY scripts/11_abliterate.py --model $A --name sm_abl_calc --domain calc --limit 4 --max-new 256
  run abl_book $PY scripts/11_abliterate.py --model $A --name sm_abl_book --domain book --limit 4 --layers 6
fi
if want local_abl; then
  run local_abl $PY scripts/17_local_abl.py --model $A --name sm_labl --domain calc --layers 4,5,6 --sites resid,mlp,attn \
    --ref $T --ref-label base --n-prompts 8 --n-calc 4 --n-retain 4 --max-new 256 --gen-bs 8
fi
if want weight; then
  run weight $PY scripts/20_weight_attacks.py --model $A --name sm_wa --domain calc --noise 0.01 --n-calc 4 --n-retain 4 --max-new 256
fi

# ------------------------------------------------------------------ UNDO
if want undo; then
  stage "FineWeb shard for UNDO"
  $PY - <<'EOF' || echo "SMOKE FAIL fineweb shard"
from huggingface_hub import HfApi, hf_hub_download
import os, shutil
files = sorted(f for f in HfApi().list_repo_files("HuggingFaceFW/fineweb", repo_type="dataset") if f.startswith("sample/10BT/"))
p = hf_hub_download("HuggingFaceFW/fineweb", files[-1], repo_type="dataset", local_dir="work/fw")
shutil.move(p, "work/fineweb_shard.parquet")
print("fineweb shard", files[-1], os.path.getsize("work/fineweb_shard.parquet") // 2**20, "MB")
EOF
  run undo_calc $PY scripts/15_undo.py --teacher $A --name sm_undo_calc --target calculus --out models/sm_undo_calc \
    --fineweb work/fineweb_shard.parquet --log results/undo/sm_undo_calc/train.json --max-tokens 3e5 --max-minutes 3 \
    --step-tokens 8192 --eval-docs 4 --eval-every 10
  rm -rf models/sm_undo_calc
  run undo_book $PY scripts/15_undo.py --teacher $A --name sm_undo_book --target book --out models/sm_undo_book \
    --fineweb work/fineweb_shard.parquet --log results/undo/sm_undo_book/train.json --max-tokens 1e5 --max-minutes 2 \
    --step-tokens 8192 --eval-docs 2 --eval-every 5 --exclude-calculus
  rm -rf models/sm_undo_book
fi

# ------------------------------------------------------------------ surgery selftest
if want selftest; then
  mkdir -p work/surgery
  cp work/selfgen.jsonl work/surgery/selfgen_v3.jsonl 2>/dev/null
  cp work/selfgen_dev.jsonl work/surgery/selfgen_dev_v3.jsonl 2>/dev/null
  run selftest_cpu $PY scripts/16_surgery.py selftest
  run selftest_model $PY scripts/16_surgery.py selftest --model
fi

# ------------------------------------------------------------------ J-lens
if want jspace; then
  cp work/fineweb_sample.jsonl work/fineweb.jsonl
  $PY -m lfm3u.storage get "$BUCKET_URI/jspace/inputs/knowmem_forget_qa_jspace.jsonl" work/knowmem.jsonl
  run jspace $PY scripts/05_jspace.py --name _smoke --model $T --corpus work/fineweb.jsonl --questions work/knowmem.jsonl \
    --max-prompts 2 --min-prompts 1 --chunk 2 --out results/jspace --lens-dir work/smoke
  if ls results/jspace/_smoke/*.error.txt >/dev/null 2>&1; then
    echo "SMOKE FAIL jspace stages: $(ls results/jspace/_smoke/*.error.txt)"
    cat results/jspace/_smoke/*.error.txt
    FAILS="$FAILS jspace_stage"
  fi
fi

stage "smoke done; failed:${FAILS:- none}"
cat results/smoke/timings.txt
sync_up
