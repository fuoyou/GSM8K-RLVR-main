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
LR=3e-5
BS=2
GA=6
NG=6
MAX_LEN=300
SAVE_STEPS=100

TS="$(date +%Y%m%d_%H%M%S)"
ROOT_OUT="$SCRIPT_DIR/outputs/reward-ablation-${TS}"
LOG_DIR="$SCRIPT_DIR/logs/reward-ablation-${TS}"
mkdir -p "$ROOT_OUT" "$LOG_DIR"

SUMMARY_CSV="$LOG_DIR/summary.csv"
echo "exp_id,fmt_w,corr_w,tol,fmt_mode,train_status,final_checkpoint,eval_status,final_accuracy,train_log,eval_log" > "$SUMMARY_CSV"

find_last_checkpoint() {
  local out_dir="$1"
  ls -1 "$out_dir" 2>/dev/null | sed -n 's/^\(checkpoint-[0-9][0-9]*\)$/\1/p' | sort -t- -k2 -n | tail -n 1
}

extract_final_accuracy() {
  local eval_log="$1"
  sed -n 's/^Final Accuracy: //p' "$eval_log" | tail -n 1
}

run_one() {
  local exp_id="$1"
  local fmt_w="$2"
  local corr_w="$3"
  local tol="$4"
  local fmt_mode="$5"
  local run_name="Qwen2.5-Math-1.5B_reward_${exp_id}_fw${fmt_w}_cw${corr_w}_tol${tol}_mode${fmt_mode}"
  local out_dir="$ROOT_OUT/$run_name"
  local train_log="$LOG_DIR/${exp_id}_train.log"
  local eval_log="$LOG_DIR/${exp_id}_eval.log"
  local eval_pred="$LOG_DIR/${exp_id}_preds.jsonl"
  local train_status="done"
  local eval_status="not_run"
  local final_ckpt="-"
  local final_acc="-"

  mkdir -p "$out_dir"
  echo "[RUN] $exp_id fw=$fmt_w cw=$corr_w tol=$tol mode=$fmt_mode" | tee -a "$LOG_DIR/master.log"

  if ! python train.py \
    --model_name "$BASE_MODEL" \
    --output_dir "$out_dir" \
    --attn_implementation sdpa \
    --report_to none \
    --num_shots 2 \
    --learning_rate "$LR" \
    --per_device_train_batch_size "$BS" \
    --gradient_accumulation_steps "$GA" \
    --num_generations "$NG" \
    --max_completion_length "$MAX_LEN" \
    --max_steps "$MAX_STEPS" \
    --seed "$SEED" \
    --save_steps "$SAVE_STEPS" \
    --reward_format_weight "$fmt_w" \
    --reward_correctness_weight "$corr_w" \
    --reward_correctness_tolerance "$tol" \
    --reward_format_mode "$fmt_mode" \
    > "$train_log" 2>&1; then
    train_status="failed"
  fi

  if [[ "$train_status" == "done" ]]; then
    final_ckpt="$(find_last_checkpoint "$out_dir" || true)"
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
        final_acc="$(extract_final_accuracy "$eval_log")"
      else
        eval_status="failed"
      fi
    else
      train_status="no_checkpoint"
    fi
  fi

  echo "${exp_id},${fmt_w},${corr_w},${tol},${fmt_mode},${train_status},${final_ckpt},${eval_status},\"${final_acc}\",${train_log},${eval_log}" >> "$SUMMARY_CSV"
}

run_one "rw-001" "0.5" "1.0" "1e-3" "strict"
run_one "rw-002" "0.2" "1.0" "1e-3" "strict"
run_one "rw-003" "0.8" "1.0" "1e-3" "strict"
run_one "rw-004" "0.5" "1.2" "1e-3" "strict"
run_one "rw-005" "0.5" "1.0" "1e-2" "strict"
run_one "rw-006" "0.5" "1.0" "1e-3" "loose"

echo "[DONE] reward ablation complete. summary=$SUMMARY_CSV" | tee -a "$LOG_DIR/master.log"
