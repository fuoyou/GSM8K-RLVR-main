#!/usr/bin/env bash
# Quick GRPO run for fast usable LoRA checkpoint.
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
REPORT_TO="${REPORT_TO:-none}"

# Keep generation_batch_size divisible by num_generations:
# per_device_train_batch_size(1) * gradient_accumulation_steps(2) == num_generations(2)
exec python train.py \
  --model_name "$MODEL_PATH" \
  --attn_implementation sdpa \
  --report_to "$REPORT_TO" \
  --num_shots 1 \
  --max_steps 300 \
  --learning_rate 1e-5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 2 \
  --num_generations 2 \
  --max_completion_length 128 \
  --save_steps 50 \
  "$@"
