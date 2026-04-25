#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/root/miniconda3/etc/profile.d/conda.sh"
fi
conda activate gsm8k-rlvr

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

BASE_MODEL="${GSM8K_MODEL:-$SCRIPT_DIR/models/Qwen2.5-Math-1.5B}"
SEED=42
NUM_SHOTS=8
MAX_NEW_TOKENS=400
EVAL_LIMIT=200
MAX_STEPS=300
GA=6
NG=6
MAX_LEN=300
SAVE_STEPS=100

TS="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="$SCRIPT_DIR/outputs/phaseA-grid-${TS}"
LOG_DIR="$SCRIPT_DIR/logs/phaseA-grid-${TS}"
mkdir -p "$ROOT_OUT" "$LOG_DIR"

SUMMARY_CSV="$LOG_DIR/summary.csv"
echo "exp_id,lr,bs,ga,ng,max_len,max_steps,seed,train_status,final_checkpoint,eval_status,final_accuracy,train_log,eval_log" > "$SUMMARY_CSV"

run_one() {
  local exp_id="$1"
  local lr="$2"
  local bs="$3"
  local run_name="Qwen2.5-Math-1.5B_lr${lr}_bs${bs}_ga${GA}_ng${NG}_steps${MAX_STEPS}_seed${SEED}"
  local out_dir="$ROOT_OUT/$run_name"
  local train_log="$LOG_DIR/${exp_id}_train.log"
  local eval_log="$LOG_DIR/${exp_id}_eval.log"
  local eval_pred="$LOG_DIR/${exp_id}_preds.jsonl"
  local train_status="done"
  local eval_status="not_run"
  local final_ckpt="-"
  local final_acc="-"

  mkdir -p "$out_dir"
  echo "[RUN] $exp_id lr=$lr bs=$bs out=$out_dir" | tee -a "$LOG_DIR/master.log"

  if ! python train.py \
    --model_name "$BASE_MODEL" \
    --output_dir "$out_dir" \
    --attn_implementation sdpa \
    --report_to none \
    --num_shots 2 \
    --learning_rate "$lr" \
    --per_device_train_batch_size "$bs" \
    --gradient_accumulation_steps "$GA" \
    --num_generations "$NG" \
    --max_completion_length "$MAX_LEN" \
    --max_steps "$MAX_STEPS" \
    --seed "$SEED" \
    --save_steps "$SAVE_STEPS" \
    > "$train_log" 2>&1; then
    train_status="failed"
  fi

  if [[ "$train_status" == "done" ]]; then
    final_ckpt="$(ls -1 "$out_dir" | rg '^checkpoint-' | sort -t- -k2 -n | tail -n 1 || true)"
    if [[ -n "$final_ckpt" ]]; then
      if python eval_gsm8k.py \
        --base_model "$BASE_MODEL" \
        --adapter_path "$out_dir/$final_ckpt" \
        --batch_size 1 \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --num_shots "$NUM_SHOTS" \
        --seed "$SEED" \
        --limit "$EVAL_LIMIT" \
        --save_predictions "$eval_pred" \
        > "$eval_log" 2>&1; then
        eval_status="done"
        final_acc="$(rg "Final Accuracy:" "$eval_log" -o -r '$0' | tail -n 1 | sed 's/Final Accuracy: //')"
      else
        eval_status="failed"
      fi
    else
      train_status="no_checkpoint"
    fi
  fi

  echo "${exp_id},${lr},${bs},${GA},${NG},${MAX_LEN},${MAX_STEPS},${SEED},${train_status},${final_ckpt},${eval_status},\"${final_acc}\",${train_log},${eval_log}" >> "$SUMMARY_CSV"
}

run_one "exp-001" "1e-5" "1"
run_one "exp-002" "1e-5" "2"
run_one "exp-003" "2e-5" "1"
run_one "exp-004" "2e-5" "2"
run_one "exp-005" "3e-5" "1"
run_one "exp-006" "3e-5" "2"

echo "[DONE] phaseA grid complete. summary=$SUMMARY_CSV" | tee -a "$LOG_DIR/master.log"
