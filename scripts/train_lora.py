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
import time
import json
import math
from pathlib import Path

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if platform.system() == "Linux":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from utils.dataset_checks import assert_disjoint
from utils.training_status import StatusWriter
from utils.active_dataset import active_release
from utils.training_checkpoints import seal_checkpoint, latest_checkpoint, valid_checkpoint, digest

try:
    import psutil
except ImportError:
    psutil = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
DEFAULT_RELEASE = active_release()
DEFAULT_TRAIN_FILE = DEFAULT_RELEASE / "train/train_combined.jsonl"
DEFAULT_VAL_FILE = DEFAULT_RELEASE / "validation/validation_combined.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / ("checkpoints-" + DEFAULT_RELEASE.name.replace('_', '-'))


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


class LiveStatusCallback(TrainerCallback):
    """Report trainer state and cooperatively stop at optimizer-step boundaries."""
    def __init__(self, writer, stop_file=None, checkpoint_steps=None):
        self.writer = writer
        self.stop_file = Path(stop_file) if stop_file else None
        self.stopped = False
        self.last_evaluation_update = 0
        self.start_step = 0
        self.checkpoint_steps = checkpoint_steps

    def publish(self, state, phase="training", **values):
        memory = {}
        if torch.cuda.is_available():
            memory = {"gpu": torch.cuda.get_device_name(0),
                      "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
                      "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)}
        if psutil:
            memory["ram_gb"] = round(psutil.Process().memory_info().rss / 1024**3, 2)
        elapsed = time.monotonic() - self.writer.started
        done = state.global_step - self.start_step
        eta = elapsed / done * (state.max_steps - state.global_step) if done > 0 else None
        report = dict(status=phase, step=state.global_step, max_steps=state.max_steps,
                      epoch=state.epoch, eta_seconds=eta, best_metric=state.best_metric,
                      best_checkpoint=state.best_model_checkpoint, **memory)
        report.update(values)
        self.writer.update(**report)

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_step = state.global_step
        self.publish(state)

    def on_step_end(self, args, state, control, **kwargs):
        if self.checkpoint_steps and state.global_step % self.checkpoint_steps == 0:
            control.should_save = True
        if self.stop_file and self.stop_file.exists():
            self.stopped = True
            control.should_training_stop = True
            control.should_save = True
        self.publish(state, "stopping" if self.stopped else "training")
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        values = dict(logs or {})
        if "loss" in values:
            values["loss_step"] = state.global_step
        self.publish(state, "stopping" if self.stopped else "training", **values)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.publish(state, "stopping" if self.stopped else "training", **(metrics or {}))

    def on_prediction_step(self, args, state, control, **kwargs):
        if time.monotonic() - self.last_evaluation_update >= 1:
            self.publish(state, "evaluating")
            self.last_evaluation_update = time.monotonic()

    def on_save(self, args, state, control, **kwargs):
        seal_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}")
        self.publish(state, "stopping" if self.stopped else "training", last_checkpoint=str(Path(args.output_dir) / f"checkpoint-{state.global_step}"))


def get_training_precision() -> tuple[torch.dtype, bool, bool]:
    """Use BF16 when available (e.g. Blackwell / Ada Lovelace / Ampere), otherwise FP16."""
    if not torch.cuda.is_available():
        return torch.float32, False, False

    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, True, False

    return torch.float16, False, True


def build_training_arguments(**kwargs) -> SFTConfig:
    """Create SFTConfig with options supported by the installed TRL version."""
    supported_args = inspect.signature(SFTConfig.__init__).parameters
    compatible_kwargs = {
        key: value for key, value in kwargs.items() if key in supported_args
    }
    skipped = sorted(set(kwargs) - set(compatible_kwargs))

    if skipped:
        raise RuntimeError(f"Installed TRL lacks required training options: {', '.join(skipped)}")

    return SFTConfig(**compatible_kwargs)


def get_optimizer_name() -> str:
    if torch.cuda.is_available():
        return "adamw_torch_fused"
    return "adamw_torch"


def training_schedule(args, num_examples):
    """Single-process schedule: regular best-model saves plus frequent recovery saves."""
    recovery_steps = min(args.save_steps, args.max_steps) if args.max_steps > 0 else args.save_steps
    evaluation_steps = (max(recovery_steps, min(recovery_steps * 10, args.max_steps) // recovery_steps * recovery_steps)
                        if args.max_steps > 0 else recovery_steps * 10)
    batches = math.ceil(num_examples / args.batch_size)
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * math.ceil(batches / args.grad_accum))
    return dict(eval_strategy='steps', eval_steps=evaluation_steps,
                save_strategy='steps', save_steps=evaluation_steps,
                load_best_model_at_end=True, warmup_steps=math.ceil(total_steps * .03))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Burmese Qwen LoRA Adapter")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Path to base model")
    parser.add_argument("--train_file", type=str, default=str(DEFAULT_TRAIN_FILE), help="Path to training jsonl")
    parser.add_argument("--val_file", type=str, default=str(DEFAULT_VAL_FILE), help="Path to validation jsonl")
    parser.add_argument("--test_file", type=str, default=str(DEFAULT_RELEASE / "test/test_combined.jsonl"), help="Held-out test file for leakage checks")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR), help="Output checkpoint directory")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per device batch size")
    parser.add_argument("--grad_accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for full epochs)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Bound evaluation VRAM as well as training VRAM")
    parser.add_argument("--early_stopping_patience", type=int, default=1, help="Stop after this many unimproved evaluations")
    parser.add_argument("--status_file", default=None, help="Atomic live status JSON for the desktop UI")
    parser.add_argument("--stop_file", default=None, help="Stop safely when this file appears")
    parser.add_argument("--save_steps", type=int, default=25, help="Checkpoint every N optimizer steps")
    parser.add_argument("--resume_from_checkpoint", default=None, help="Verified checkpoint directory, or latest")
    args = parser.parse_args()
    if args.save_steps < 1 or args.save_total_limit < 2:
        parser.error('save_steps must be positive and save_total_limit must be at least 2')
    output_dir = Path(args.output_dir)
    resume = None
    if args.resume_from_checkpoint:
        resume = latest_checkpoint(output_dir) if args.resume_from_checkpoint == 'latest' else Path(args.resume_from_checkpoint).resolve()
        if resume.parent.resolve() != output_dir.resolve() or not valid_checkpoint(resume):
            raise ValueError('Resume requires a verified checkpoint inside output_dir.')
    contract = {k: getattr(args,k) for k in ('model_path','epochs','batch_size','grad_accum','learning_rate','lora_r','lora_alpha','max_seq_length','max_steps')}
    contract['dataset_hashes'] = {k:digest(getattr(args,k)) for k in ('train_file','val_file','test_file')}
    contract_path = output_dir / 'resume_contract.json'
    if resume:
        if not contract_path.exists() or json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('Dataset or training settings changed; restore original settings/data to resume.')
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError('Output directory is not empty. Resume explicitly or use a new directory.')
        output_dir.mkdir(parents=True, exist_ok=True)
        contract_path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    status = StatusWriter(args.status_file)
    status.update(status="preparing", output_dir=args.output_dir)
    live = LiveStatusCallback(status, args.stop_file, checkpoint_steps=args.save_steps)

    def report_failure(exc_type, exc, traceback):
        status.update(status="failed", error=str(exc))
        sys.__excepthook__(exc_type, exc, traceback)
    sys.excepthook = report_failure

    train_path = Path(args.train_file)
    if not train_path.exists():
        print(f"ERROR: Training file not found: {train_path}")
        print("Restore the dataset release selected by datasets/active.json; see datasets/README.md.")
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
    # Preserve conversational structure so TRL can mask user/system tokens correctly.

    val_path = Path(args.val_file)
    if val_path.exists():
        print(f"Loading validation dataset from {val_path}...")
        val_dataset = load_dataset("json", data_files=str(val_path))
        train_split = dataset["train"]
        eval_split = val_dataset["train"]
    else:
        raise FileNotFoundError("Validation file missing. Build a grouped dataset split first.")

    test_path = Path(args.test_file)
    test_split = load_dataset("json", data_files=str(test_path))["train"] if test_path.exists() else []
    assert_disjoint(train_split, eval_split, test_split)

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
    status.update(status="loading model", train_samples=len(train_split), validation_samples=len(eval_split))
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
        **training_schedule(args, len(train_split)),
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=torch.cuda.is_available(),
        optim=get_optimizer_name(),
        max_length=args.max_seq_length,
        packing=False,
        assistant_only_loss=True,
        loss_type="chunked_nll",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        dataloader_num_workers=0,
        dataloader_pin_memory=True,
        logging_steps=10,
        eval_accumulation_steps=1,
        save_total_limit=args.save_total_limit,  # Automatically limits saved checkpoints!
        restore_callback_states_from_checkpoint=True,
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
        callbacks=[MemoryTraceCallback(), EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience), live],
    )

    print("\n" + "=" * 70)
    print("STARTING BURMESE LORA TRAINING")
    print(f"  Total Epochs:          {args.epochs}")
    print(f"  Batch Size (Effective):{args.batch_size * args.grad_accum}")
    print(f"  Learning Rate:         {args.learning_rate}")
    print(f"  LoRA Rank (r):         {args.lora_r} (alpha={args.lora_alpha})")
    print(f"  Checkpoint Limit:      {args.save_total_limit} (auto-pruned)")
    print("=" * 70 + "\n")

    training_result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
    status.update(status="saving")
    trainer.save_state()
    trainer.log_metrics("train", training_result.metrics)
    trainer.save_metrics("train", training_result.metrics)
    (output_dir / "training_manifest.json").write_text(json.dumps({
        "model_path": args.model_path, "train_file": str(train_path), "val_file": str(val_path),
        "test_file": str(test_path), "assistant_only_loss": True,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric, "global_step": trainer.state.global_step,
        "arguments": vars(args),
        "metrics": training_result.metrics, "stopped_by_user": live.stopped,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSaving final model & tokenizer to {output_dir}...")
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)
    status.update(status="stopped" if live.stopped else "completed", metrics=training_result.metrics,
                  adapter_path=str(output_dir), step=trainer.state.global_step,
                  best_checkpoint=trainer.state.best_model_checkpoint)
    print("Training stopped safely; adapter saved." if live.stopped else "Training successfully completed!")


if __name__ == "__main__":
    main()
