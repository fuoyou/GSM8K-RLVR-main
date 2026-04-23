#!/usr/bin/env bash
# Resume full training from latest checkpoint in fresh-output directory.
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

MODEL_PATH="${GSM8K_MODEL:-$SCRIPT_DIR/models/Qwen2.5-Math-1.5B}"
OUT_DIR="${GSM8K_OUTPUT_DIR:-$SCRIPT_DIR/outputs/GRPO-full-from-scratch/qa/Qwen2.5-Math-1.5B}"
REPORT_TO="${REPORT_TO:-none}"

LATEST_CKPT="$(ls -d "$OUT_DIR"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "No checkpoint found under: $OUT_DIR"
  echo "Run ./run_train_full_fresh.sh first."
  exit 1
fi

echo "Resuming from: $LATEST_CKPT"
exec python train.py \
  --model_name "$MODEL_PATH" \
  --output_dir "$OUT_DIR" \
  --attn_implementation sdpa \
  --report_to "$REPORT_TO" \
  --num_shots 2 \
  --learning_rate 2e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 6 \
  --num_generations 6 \
  --max_completion_length 300 \
  --num_train_epochs 2 \
  --save_steps 100 \
  --resume_from_checkpoint "$LATEST_CKPT" \
  "$@"
