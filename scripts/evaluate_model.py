"""Evaluate the LoRA model against the prompt set."""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.generation_utils import GENERATION_CONFIG
from utils.model_utils import get_gpu_info


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
LORA_PATH = PROJECT_ROOT / "outputs" / "checkpoints"
PROMPTS_FILE = PROJECT_ROOT / "prompts" / "evaluation_prompts.json"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluations" / "results"
VALID_MODES = {"--think", "--nothink"}


def get_inference_dtype() -> torch.dtype | str:
    if not torch.cuda.is_available():
        return "auto"
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_prompts() -> list[dict | str]:
    with PROMPTS_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def parse_mode(argv: list[str]) -> str:
    if len(argv) < 2:
        raise SystemExit("Usage: python evaluate_model.py --think|--nothink")

    mode = argv[1]
    if mode not in VALID_MODES:
        raise SystemExit("Invalid mode. Use --think or --nothink")

    return mode


def main() -> None:
    import sys

    mode = parse_mode(sys.argv)
    prefix = "/no_think\n" if mode == "--nothink" else ""

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    date = now.strftime("%Y%m%d")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=get_inference_dtype(),
        attn_implementation="sdpa",
        device_map="auto",
    )

    print("Loading LoRA...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)
    model.eval()

    gpu = get_gpu_info()
    report = [
        "=" * 80,
        "BEACON EVALUATION REPORT",
        "=" * 80,
        f"Time: {timestamp}",
        f"Model: {MODEL_PATH}",
        f"Mode: {mode}",
        f"GPU: {gpu['gpu_name']}\n",
    ]

    total_time = 0.0
    total_tokens = 0

    print("\n=== EVALUATION START ===\n")

    with torch.inference_mode():
        for index, item in enumerate(prompts, 1):
            prompt_text = item["prompt"] if isinstance(item, dict) else item
            category = item.get("category", "uncategorized") if isinstance(item, dict) else "uncategorized"
            prompt = prefix + prompt_text

            chat_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
            prompt_tokens = inputs.input_ids.shape[1]

            start = time.perf_counter()
            output = model.generate(**inputs, **GENERATION_CONFIG)
            elapsed = time.perf_counter() - start

            generated = output[0][prompt_tokens:]
            generated_tokens = len(generated)
            response = tokenizer.decode(generated, skip_special_tokens=True)

            total_time += elapsed
            total_tokens += generated_tokens
            tokens_per_second = generated_tokens / elapsed if elapsed else 0.0

            print(f"[{index}/{len(prompts)}] {generated_tokens} tok | {elapsed:.2f}s")

            report.extend(
                [
                    "=" * 80,
                    f"TEST {index} | {category}",
                    f"Prompt: {prompt}",
                    f"Response: {response}",
                    f"Tokens: {generated_tokens} | Time: {elapsed:.2f}s | TPS: {tokens_per_second:.2f}\n",
                ]
            )

    report.extend(
        [
            "=" * 80,
            "SUMMARY",
            "=" * 80,
            f"Prompts: {len(prompts)}",
            f"Total Tokens: {total_tokens}",
            f"Total Time: {total_time:.2f}s",
            f"Avg TPS: {(total_tokens / total_time) if total_time else 0.0:.2f}",
        ]
    )

    report_file = RESULTS_DIR / f"e{date}.txt"
    report_file.write_text("\n".join(report), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"Report saved: {report_file}")


if __name__ == "__main__":
    main()
