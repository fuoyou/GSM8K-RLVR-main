#!/usr/bin/env bash
# GSM8K-RLVR: use dedicated conda env (isolated from other /data projects).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "/root/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/root/miniconda3/etc/profile.d/conda.sh"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found; activate your own env with Python 3.10+ and torch, then: python train.py ..."
  exit 1
fi

conda activate gsm8k-rlvr

export TOKENIZERS_PARALLELISM=false

# --- Hub download host (模型权重走哪台服务器下载) ---
# 默认是 https://huggingface.co；若该域名 DNS 不通，可改成你能访问的镜像（须兼容 HF Hub API）：
#   export HF_ENDPOINT=https://hf-mirror.com
# 其它常见写法见各镜像站说明；以你网络能解析、能 HTTPS 访问为准。

# --- 缓存目录（下载下来的模型放哪，不是“换模型”，只是换磁盘路径）---
#   export HF_HUB_CACHE=/data/hf-hub-cache

# --- 完全不经过 Hub：把模型文件夹拷到本机后 ---
#   python train.py --model_name /data/models/Qwen2.5-Math-1.5B
# 目录内需含 config.json、tokenizer 与权重分片等。

# Optional: match original logging
# export WANDB_API_KEY=... && REPORT_TO=wandb

REPORT_TO="${REPORT_TO:-none}"
# Local base model (downloaded via hf-mirror curl); override with GSM8K_MODEL=...
MODEL_PATH="${GSM8K_MODEL:-$SCRIPT_DIR/models/Qwen2.5-Math-1.5B}"

exec python train.py --report_to "$REPORT_TO" --model_name "$MODEL_PATH" "$@"
