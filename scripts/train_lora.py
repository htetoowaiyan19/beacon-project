"""Train the Qwen model with LoRA adapters.

The defaults target a 16 GB GPU with full bf16/fp16 LoRA: bounded sequence
length, gradient checkpointing, and a small validation split for quality checks.
"""

from __future__ import annotations

import os
import platform

# Reduces CUDA allocator fragmentation from variable-length batches.
# Linux-only: the CUDA allocator on Windows doesn't support this feature
# and will just print a harmless "not supported on this platform" warning
# if set there, so skip it entirely on Windows.
if platform.system() == "Linux":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import inspect
from pathlib import Path

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
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
TRAIN_FILE = PROJECT_ROOT / "datasets" / "raw" / "train" / "train7252026812026511.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

EPOCHS = 3
PER_DEVICE_BATCH_SIZE = int(os.getenv("TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "4"))
MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048"))
EVAL_SPLIT_SIZE = float(os.getenv("EVAL_SPLIT_SIZE", "0.1"))
DATASET_NUM_PROC = min(2, os.cpu_count() or 1)
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "0") == "1"
# Packing concatenates multiple examples into one sequence for efficiency,
# but without a Flash Attention 2/3 implementation the attention mask does
# not reliably prevent tokens from one packed example attending into its
# neighbor across the seam (TRL warns about this at runtime). Since the
# base model here isn't using flash_attention_2/3 by default, packing is
# OFF by default to avoid silently contaminating training. Only turn this
# on if USE_FLASH_ATTENTION=1 is also set and flash-attn actually installs
# and loads successfully for your GPU.
PACKING = os.getenv("PACKING", "0") == "1"


def check_gpu_kernel_support() -> None:
    """Warn loudly if PyTorch has no compiled kernels for this GPU.

    New GPU generations (e.g. Blackwell / sm_120) are sometimes ahead of
    the PyTorch wheels available at install time. When that happens CUDA
    ops can fall back to slow runtime PTX JIT compilation instead of
    precompiled kernels -- training still runs and loss still goes down,
    but each step can be 10-50x slower with no obvious error. Catching
    this early saves hours of confused debugging.
    """

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    supported = torch.cuda.get_arch_list()

    if arch not in supported:
        print(
            f"WARNING: {torch.cuda.get_device_name(0)} reports compute "
            f"capability {arch}, which is NOT in this PyTorch build's "
            f"compiled kernel list ({supported}). Training may silently "
            "fall back to slow PTX JIT compilation. Consider reinstalling "
            "PyTorch from https://pytorch.org/get-started/locally/ with a "
            "CUDA version that lists your GPU's architecture."
        )


class MemoryTraceCallback(TrainerCallback):
    """Logs host RSS + CUDA memory at each logging step and at eval
    boundaries, so we can see *where* memory grows instead of guessing.

    Prints a line per event; pipe stdout to a file and plot/eyeball it
    after a short run. Look for:
      - Steady climb during TRAIN steps only -> leak in the train loop
        (e.g. something holding refs to logits/activations across steps).
      - Jumps that line up with "EVAL START"/"EVAL END" -> eval loop is
        still accumulating something in host memory.
      - RSS climbs but plateaus/drops after a GC pause -> not a real leak,
        just delayed garbage collection; harmless.
    """

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


def build_training_arguments(**kwargs) -> SFTConfig:
    """Create SFTConfig with options supported by the installed TRL version."""

    supported_args = inspect.signature(SFTConfig.__init__).parameters
    compatible_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in supported_args
    }
    skipped = sorted(set(kwargs) - set(compatible_kwargs))

    if skipped:
        print(f"Skipping unsupported TrainingArguments: {', '.join(skipped)}")

    return SFTConfig(**compatible_kwargs)


def get_attention_implementation() -> str:
    if USE_FLASH_ATTENTION:
        return "flash_attention_2"

    return "sdpa"


def get_optimizer_name() -> str:
    if torch.cuda.is_available():
        return "adamw_torch_fused"

    return "adamw_torch"


def main() -> None:
    check_gpu_kernel_support()
    torch_dtype, use_bf16, use_fp16 = get_training_precision()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, model_max_length=MAX_SEQ_LENGTH)
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
    split_dataset = dataset["train"].train_test_split(
        test_size=EVAL_SPLIT_SIZE,
        seed=42,
        shuffle=True,
    )

    lora_config = LoraConfig(
        r=int(os.getenv("LORA_R", "32")),
        lora_alpha=int(os.getenv("LORA_ALPHA", "64")),
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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch_dtype,
        attn_implementation=get_attention_implementation(),
        device_map={"": 0} if torch.cuda.is_available() else None,
    )
    model.config.use_cache = False

    model = get_peft_model(model, lora_config)
    # Required for gradient checkpointing to work correctly with PEFT: since
    # only the LoRA adapter weights require grad (base model is frozen),
    # autograd needs this hook to know where to anchor the recompute graph.
    # Without it, checkpointing silently retains more activations than
    # expected, contributing to VRAM pressure.
    model.enable_input_require_grads()
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
        optim=get_optimizer_name(),
        max_length=MAX_SEQ_LENGTH,
        packing=PACKING,
        # Avoids materializing the full (batch x seq_len x vocab) logits
        # tensor every step; skips ignored/padding positions before the
        # lm_head matmul instead. Falls back silently on older TRL
        # versions via build_training_arguments' kwarg filtering.
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
        eval_strategy="epoch",
        # Flush eval logits to CPU after every eval batch instead of
        # accumulating the full (batch x seq_len x vocab) tensor on GPU
        # across the whole eval set. With a ~152k vocab this was almost
        # certainly the main driver of VRAM -> shared RAM spillover during
        # evaluation.
        eval_accumulation_steps=1,
        save_strategy="epoch",
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        args=training_args,
        processing_class=tokenizer,
        callbacks=[MemoryTraceCallback()],
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()