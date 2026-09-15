"""Train the Qwen model with LoRA adapters on Burmese conversational datasets.

The defaults target a 16 GB GPU (e.g. RTX 5060 Ti / 4080 / 4090) with full bf16/fp16 LoRA:
bounded sequence length, gradient checkpointing, chunked NLL loss, and automatic checkpoint pruning.
"""

from __future__ import annotations

import argparse
import inspect
import os
import platform
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

if platform.system() == "Linux":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

try:
    import psutil
except ImportError:
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "datasets" / "clean" / "train" / "train_combined.jsonl"
DEFAULT_VAL_FILE = PROJECT_ROOT / "datasets" / "clean" / "validation" / "validation_combined.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def check_gpu_kernel_support() -> None:
    """Warn loudly if PyTorch has no compiled kernels for this GPU."""
    if not torch.cuda.is_available():
        print("INFO: CUDA is not available, running on CPU.")
        return

    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()

    print(f"INFO: Detected GPU: {gpu_name} ({vram_gb:.1f} GB VRAM, Compute Capability: {arch})")
    if arch not in supported:
        print(
            f"WARNING: {gpu_name} reports compute capability {arch}, which is NOT "
            f"in this PyTorch build's compiled kernel list ({supported})."
        )


class MemoryTraceCallback(TrainerCallback):
    """Logs host RSS + CUDA memory at each logging step and at eval boundaries."""

    def _log(self, tag: str, step: int) -> None:
        if psutil is None:
            return
        rss_mb = psutil.Process().memory_info().rss / (1024 ** 2)
        cuda_alloc_mb = cuda_reserved_mb = 0.0
        if torch.cuda.is_available():
            cuda_alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            cuda_reserved_mb = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[memtrace] step={step:>5} {tag:<10} "
            f"rss={rss_mb:9.1f}MB cuda_alloc={cuda_alloc_mb:9.1f}MB "
            f"cuda_reserved={cuda_reserved_mb:9.1f}MB"
        )

    def on_log(self, args, state, control, **kwargs):
        self._log("train", state.global_step)

    def on_evaluate(self, args, state, control, **kwargs):
        self._log("eval_end", state.global_step)


def get_training_precision() -> tuple[torch.dtype, bool, bool]:
    """Use BF16 when available (e.g. Blackwell / Ada Lovelace / Ampere), otherwise FP16."""
    if not torch.cuda.is_available():
        return torch.float32, False, False

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False

    return torch.float16, False, True


def format_chat(example: dict, tokenizer: AutoTokenizer) -> dict[str, str]:
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


def build_training_arguments(**kwargs) -> SFTConfig:
    """Create SFTConfig with options supported by the installed TRL version."""
    supported_args = inspect.signature(SFTConfig.__init__).parameters
    compatible_kwargs = {
        key: value for key, value in kwargs.items() if key in supported_args
    }
    skipped = sorted(set(kwargs) - set(compatible_kwargs))

    if skipped:
        print(f"Skipping unsupported TrainingArguments: {', '.join(skipped)}")

    return SFTConfig(**compatible_kwargs)


def get_optimizer_name() -> str:
    if torch.cuda.is_available():
        return "adamw_torch_fused"
    return "adamw_torch"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Burmese Qwen LoRA Adapter")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to base model")
    parser.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_FILE), help="Path to training jsonl")
    parser.add_argument("--val_file", type=str, default=str(DEFAULT_VAL_FILE), help="Path to validation jsonl")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        print(f"ERROR: Training file not found: {train_path}")
        print("Please run `python scripts/build_dataset.py` first to generate the combined dataset.")
        sys.exit(1)

    check_gpu_kernel_support()
    torch_dtype, use_bf16, use_fp16 = get_training_precision()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, model_max_length=args.max_seq_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading dataset from {train_path}...")
    dataset = load_dataset("json", data_files=str(train_path))
    dataset = dataset.map(
        format_chat,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=min(2, os.cpu_count() or 1),
        remove_columns=dataset["train"].column_names,
        desc="Formatting chat samples",
    )

    val_path = Path(args.val_file)
    if val_path.exists():
        print(f"Loading validation dataset from {val_path}...")
        val_dataset = load_dataset("json", data_files=str(val_path))
        val_dataset = val_dataset.map(
            format_chat,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=min(2, os.cpu_count() or 1),
            remove_columns=val_dataset["train"].column_names,
            desc="Formatting validation samples",
        )
        train_split = dataset["train"]
        eval_split = val_dataset["train"]
    else:
        print("Validation file not found. Splitting 10% from training data...")
        split = dataset["train"].train_test_split(test_size=0.1, seed=42, shuffle=True)
        train_split = split["train"]
        eval_split = split["test"]

    print(f"Dataset summary: {len(train_split):,} train samples, {len(eval_split):,} eval samples")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    print(f"Loading base model ({torch_dtype}) from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch_dtype,
        attn_implementation="sdpa",
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = build_training_arguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=torch.cuda.is_available(),
        optim=get_optimizer_name(),
        max_length=args.max_seq_length,
        packing=False,
        loss_type="chunked_nll",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        group_by_length=True,
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        logging_steps=10,
        eval_strategy="steps" if args.max_steps > 0 else "epoch",
        eval_steps=10 if args.max_steps > 0 else None,
        eval_accumulation_steps=1,
        save_strategy="steps" if args.max_steps > 0 else "epoch",
        save_steps=10 if args.max_steps > 0 else None,
        save_total_limit=args.save_total_limit,  # Automatically limits saved checkpoints!
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_split,
        eval_dataset=eval_split,
        args=training_args,
        processing_class=tokenizer,
        callbacks=[MemoryTraceCallback()],
    )

    print("\n" + "=" * 70)
    print("STARTING BURMESE LORA TRAINING")
    print(f"  Total Epochs:          {args.epochs}")
    print(f"  Batch Size (Effective):{args.batch_size * args.grad_accum}")
    print(f"  Learning Rate:         {args.learning_rate}")
    print(f"  LoRA Rank (r):         {args.lora_r} (alpha={args.lora_alpha})")
    print(f"  Checkpoint Limit:      {args.save_total_limit} (auto-pruned)")
    print("=" * 70 + "\n")

    trainer.train()

    print(f"\nSaving final model & tokenizer to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    print("Training successfully completed!")


if __name__ == "__main__":
    main()