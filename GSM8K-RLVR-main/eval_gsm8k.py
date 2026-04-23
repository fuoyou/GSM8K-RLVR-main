import argparse
import json
import math
import os
import re
from typing import List, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from gsm8k import GSM8K


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LoRA model on GSM8K.")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/data/GSM8K-RLVR-main/models/Qwen2.5-Math-1.5B",
        help="Base model directory or hub id.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="Optional LoRA adapter directory. If not set, evaluate base model only.",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_shots", type=int, default=8)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for deterministic few-shot exemplar sampling.",
    )
    parser.add_argument("--limit", type=int, default=-1, help="Evaluate first N samples; -1 means full test set.")
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Optional progress JSON path for resume support.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save progress every N batches when --resume_from is set.",
    )
    parser.add_argument(
        "--save_predictions",
        type=str,
        default=None,
        help="Optional JSONL path to save per-sample predictions for analysis.",
    )
    return parser.parse_args()


def parse_answer_number(text: str) -> Optional[float]:
    match = re.search(r"####.*?([\-]?[\d,]+(?:\.\d+)?)", text)
    if not match:
        return None
    value = match.group(1).replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(value)
    except ValueError:
        return None


def is_correct(pred: Optional[float], gt: Optional[float], tol: float = 1e-3) -> bool:
    if pred is None or gt is None:
        return False
    return math.isclose(pred, gt, rel_tol=0.0, abs_tol=tol)


def batched(seq: List[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n], i


def load_progress(progress_path: str):
    if not os.path.exists(progress_path):
        return 0, 0
    with open(progress_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("next_index", 0)), int(data.get("correct", 0))


def save_progress(progress_path: str, next_index: int, correct: int, total: int):
    payload = {
        "next_index": next_index,
        "correct": correct,
        "total": total,
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2)


def main():
    args = parse_args()
    print(args)

    dataset = GSM8K(
        split="test",
        include_answer=False,
        include_reasoning=True,
        few_shot=True,
        num_shots=args.num_shots,
        seed=args.seed,
        cot=True,
        template="qa",
    ).dataset
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    print(f"Evaluating samples: {len(dataset)}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map="auto",
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(base_model, args.adapter_path)
        print(f"Loaded adapter: {args.adapter_path}")
    else:
        model = base_model
        print("No adapter provided; evaluating base model only.")
    model.eval()

    prompts = [x["prompt"] for x in dataset]
    gt_values = [parse_answer_number(f"#### {x['final_answer']}") for x in dataset]

    correct = 0
    total = len(prompts)
    resume_start_idx = 0
    prediction_lines = []

    if args.resume_from:
        resume_start_idx, correct = load_progress(args.resume_from)
        if resume_start_idx > total:
            raise ValueError(f"resume index {resume_start_idx} exceeds total samples {total}")
        print(f"Resuming from index={resume_start_idx}, recovered_correct={correct}")

    processed_batches = 0

    for batch_prompts, start_idx in batched(prompts, args.batch_size):
        if start_idx < resume_start_idx:
            continue
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = out[:, enc["input_ids"].shape[1]:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for i, text in enumerate(texts):
            idx = start_idx + i
            pred_val = parse_answer_number(text)
            sample_correct = is_correct(pred_val, gt_values[idx])
            if sample_correct:
                correct += 1
            if args.save_predictions:
                prediction_lines.append(
                    json.dumps(
                        {
                            "idx": idx,
                            "pred_text": text,
                            "pred_value": pred_val,
                            "gt_value": gt_values[idx],
                            "correct": sample_correct,
                        },
                        ensure_ascii=True,
                    )
                )

        processed_batches += 1
        done = start_idx + len(batch_prompts)
        if args.resume_from and processed_batches % max(args.save_every, 1) == 0:
            save_progress(args.resume_from, done, correct, total)

        if (start_idx // args.batch_size + 1) % 10 == 0:
            acc = 100.0 * correct / done
            print(f"Progress: {done}/{total}, running_acc={acc:.2f}%")

    final_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"Final Accuracy: {final_acc:.2f}% ({correct}/{total})")
    if args.save_predictions:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_predictions)), exist_ok=True)
        with open(args.save_predictions, "w", encoding="utf-8") as f:
            f.write("\n".join(prediction_lines))
            if prediction_lines:
                f.write("\n")
        print(f"Saved predictions: {args.save_predictions}")
    if args.resume_from:
        save_progress(args.resume_from, total, correct, total)


if __name__ == "__main__":
    main()
