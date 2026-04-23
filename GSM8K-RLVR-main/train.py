import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils import format_reward_func_qa, correctness_reward_func_qa, \
                  format_reward_func_code, correctness_reward_func_code, \
                  print_trainable_parameters
from gsm8k import GSM8K


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for Reasoning on GSM8K with RLVR and GRPO.")
    parser.add_argument('--format', type=str, default='qa', choices=['qa', 'code'])
    parser.add_argument('--output_dir', type=str, default=None, help="Optional output directory override.")
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed for deterministic few-shot exemplar sampling and train-set shuffling.",
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='Qwen/Qwen2.5-Math-1.5B',
        help="Hub repo id (e.g. Qwen/...) or local directory with config.json. Hub base URL: HF_ENDPOINT env (default huggingface.co).",
    )
    parser.add_argument(
        '--attn_implementation',
        type=str,
        default='auto',
        choices=['auto', 'flash_attention_2', 'sdpa', 'eager'],
        help="Attention backend. 'auto' tries flash_attention_2 then falls back to sdpa.",
    )
    parser.add_argument(
        '--report_to',
        type=str,
        default='none',
        choices=['none', 'wandb', 'tensorboard'],
        help="Logging backend. Use 'wandb' to match the original repo (requires WANDB login).",
    )
    parser.add_argument(
        '--max_steps',
        type=int,
        default=-1,
        help="If > 0, run only this many optimizer steps (overrides num_train_epochs; useful for smoke tests).",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint folder for resuming training.",
    )
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--per_device_train_batch_size', type=int, default=1)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=6)
    parser.add_argument('--num_generations', type=int, default=6)
    parser.add_argument('--max_completion_length', type=int, default=300)
    parser.add_argument('--num_train_epochs', type=float, default=2.0)
    parser.add_argument('--save_steps', type=int, default=100)
    return parser.parse_args()


def load_causal_lm(model_name: str, attn: str):
    """Load model; when attn is 'auto', try flash_attention_2 then sdpa."""
    if attn == "auto":
        candidates = ("flash_attention_2", "sdpa")
    else:
        candidates = (attn,)
    last_err = None
    for impl in candidates:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                attn_implementation=impl,
                device_map="auto",
            )
            print(f"Using attn_implementation={impl}")
            return model
        except Exception as exc:
            last_err = exc
            print(f"attn_implementation={impl} failed: {exc}")
    raise last_err


def main():
    args = parse_args()
    print(args)

    dataset = GSM8K(
        split='train',
        include_answer=False,
        include_reasoning=True,
        few_shot=True,
        num_shots=args.num_shots,
        seed=args.seed,
        cot=True,
        template=args.format,
    ).dataset.shuffle(seed=args.seed)

    model_name = args.model_name
    model_short = model_name.split("/")[-1]
    output_dir = args.output_dir or os.path.join("outputs", "GRPO", args.format, model_short.replace(os.sep, "_"))

    model = load_causal_lm(model_name, args.attn_implementation)

    grpo_kwargs = dict(
        output_dir=output_dir,
        run_name=f"GRPO-GSM8K-{args.format}-{model_short}",
        learning_rate=args.learning_rate,
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        # TRL>=1.2: generation_batch_size = per_device * steps_per_generation must divide num_generations.
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        max_grad_norm=0.1,
        report_to=args.report_to,
        log_on_each_node=False,
    )
    if args.max_steps > 0:
        grpo_kwargs["max_steps"] = args.max_steps
    training_args = GRPOConfig(**grpo_kwargs)

    rank = 16
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        bias="none",
        lora_dropout=0.05,
    )

    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    if args.format == "qa":
        rewards_funcs = [format_reward_func_qa, correctness_reward_func_qa]
    elif args.format == "code":
        rewards_funcs = [format_reward_func_code, correctness_reward_func_code]
    else:
        rewards_funcs = []

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=rewards_funcs,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    model.save_pretrained(output_dir)
    print(f"LoRA model and configuration saved to {output_dir}")


if __name__ == "__main__":
    main()