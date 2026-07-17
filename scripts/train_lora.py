"""Train the Qwen model with LoRA adapters.

The defaults keep the same training quality target as the original script:
3 epochs and effective batch size 4. Speed improvements come from better GPU
precision, fused optimizer support, faster dataset preparation, and less noisy
logging rather than reducing the amount of training.
"""

from __future__ import annotations

import os
import inspect
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
TRAIN_FILE = PROJECT_ROOT / "datasets" / "raw" / "train" / "dataJUL1226a497.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

EPOCHS = 3
PER_DEVICE_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
DATASET_NUM_PROC = min(4, os.cpu_count() or 1)


def get_training_precision() -> tuple[torch.dtype, bool, bool]:
    """Use BF16 when available, otherwise use FP16 on CUDA."""

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


def build_training_arguments(**kwargs) -> TrainingArguments:
    """Create TrainingArguments with options supported by the installed version."""

    supported_args = inspect.signature(TrainingArguments.__init__).parameters
    compatible_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_args
    }
    skipped = sorted(set(kwargs) - set(compatible_kwargs))

    if skipped:
        print(f"Skipping unsupported TrainingArguments: {', '.join(skipped)}")

    return TrainingArguments(**compatible_kwargs)


def main() -> None:
    torch_dtype, use_bf16, use_fp16 = get_training_precision()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(TRAIN_FILE))
    dataset = dataset.map(
        format_chat,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=DATASET_NUM_PROC,
        remove_columns=dataset["train"].column_names,
        desc="Formatting chat samples",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch_dtype,
        attn_implementation="sdpa",
        device_map="auto",
    )
    model.config.use_cache = False

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = build_training_arguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=2e-4,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=torch.cuda.is_available(),
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        group_by_length=True,
        dataloader_num_workers=min(4, os.cpu_count() or 1),
        dataloader_pin_memory=torch.cuda.is_available(),
        logging_steps=10,
        save_strategy="epoch",
        save_safetensors=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
