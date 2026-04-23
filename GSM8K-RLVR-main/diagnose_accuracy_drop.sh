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

BASE_MODEL="${BASE_MODEL:-$SCRIPT_DIR/models/Qwen2.5-Math-1.5B}"
ADAPTER_ROOT="${ADAPTER_ROOT:-$SCRIPT_DIR/outputs/GRPO-full-from-scratch/qa/Qwen2.5-Math-1.5B}"
SEED="${SEED:-42}"
LIMIT="${LIMIT:-200}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
NUM_SHOTS="${NUM_SHOTS:-8}"
CHECKPOINTS=(${CHECKPOINTS:-"5000 7500"})
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/logs/diagnose_seed${SEED}_limit${LIMIT}}"

mkdir -p "$OUT_DIR"

echo "== Diagnose accuracy drop =="
echo "base_model: $BASE_MODEL"
echo "adapter_root: $ADAPTER_ROOT"
echo "checkpoints: ${CHECKPOINTS[*]}"
echo "seed: $SEED, limit: $LIMIT, shots: $NUM_SHOTS"
echo

for ckpt in "${CHECKPOINTS[@]}"; do
  adapter_path="$ADAPTER_ROOT/checkpoint-$ckpt"
  pred_path="$OUT_DIR/preds_ckpt${ckpt}.jsonl"
  log_path="$OUT_DIR/eval_ckpt${ckpt}.log"

  echo "[RUN] checkpoint-$ckpt"
  python eval_gsm8k.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$adapter_path" \
    --batch_size "$BATCH_SIZE" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --num_shots "$NUM_SHOTS" \
    --seed "$SEED" \
    --limit "$LIMIT" \
    --save_predictions "$pred_path" | tee "$log_path"
  echo
done

if (( ${#CHECKPOINTS[@]} >= 2 )); then
  echo "[COMPARE] first checkpoint vs second checkpoint"
  export CKPT_A="${CHECKPOINTS[0]}"
  export CKPT_B="${CHECKPOINTS[1]}"
  export OUT_DIR
  python - <<'PY'
import json
import os
import re

out_dir = os.environ["OUT_DIR"]
ckpt_a = os.environ["CKPT_A"]
ckpt_b = os.environ["CKPT_B"]

def load_preds(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda x: x["idx"])
    return rows

def has_format(text):
    return bool(re.search(r"####.*?([\-]?[\d,]+(?:\.\d+)?)", text or ""))

a = load_preds(os.path.join(out_dir, f"preds_ckpt{ckpt_a}.jsonl"))
b = load_preds(os.path.join(out_dir, f"preds_ckpt{ckpt_b}.jsonl"))

n = min(len(a), len(b))
drop = []
improve = []
for i in range(n):
    ra = a[i]
    rb = b[i]
    if ra["idx"] != rb["idx"]:
        continue
    if ra["correct"] and not rb["correct"]:
        drop.append((ra, rb))
    elif (not ra["correct"]) and rb["correct"]:
        improve.append((ra, rb))

fmt_a = sum(1 for x in a[:n] if has_format(x.get("pred_text", "")))
fmt_b = sum(1 for x in b[:n] if has_format(x.get("pred_text", "")))
acc_a = sum(1 for x in a[:n] if x["correct"]) / max(n, 1)
acc_b = sum(1 for x in b[:n] if x["correct"]) / max(n, 1)

print(f"n={n}")
print(f"ckpt{ckpt_a}: acc={acc_a*100:.2f}%, format_hit={fmt_a}/{n} ({fmt_a/max(n,1)*100:.2f}%)")
print(f"ckpt{ckpt_b}: acc={acc_b*100:.2f}%, format_hit={fmt_b}/{n} ({fmt_b/max(n,1)*100:.2f}%)")
print(f"drop(correct->wrong): {len(drop)}")
print(f"improve(wrong->correct): {len(improve)}")

report_path = os.path.join(out_dir, f"compare_ckpt{ckpt_a}_vs_ckpt{ckpt_b}.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"n={n}\n")
    f.write(f"ckpt{ckpt_a}: acc={acc_a*100:.2f}%, format_hit={fmt_a}/{n}\n")
    f.write(f"ckpt{ckpt_b}: acc={acc_b*100:.2f}%, format_hit={fmt_b}/{n}\n")
    f.write(f"drop(correct->wrong): {len(drop)}\n")
    f.write(f"improve(wrong->correct): {len(improve)}\n\n")
    f.write("Top dropped samples (first 20)\n")
    for ra, rb in drop[:20]:
        f.write(f"\nidx={ra['idx']} gt={ra['gt_value']}\n")
        f.write(f"ckpt{ckpt_a} pred={ra['pred_value']} correct={ra['correct']}\n")
        f.write(f"ckpt{ckpt_b} pred={rb['pred_value']} correct={rb['correct']}\n")
        f.write("----\n")

print(f"saved comparison: {report_path}")
PY
fi
