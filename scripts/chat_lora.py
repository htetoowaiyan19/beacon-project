"""Run a single prompt against the LoRA-adapted model."""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.generation_utils import GENERATION_CONFIG
from utils.logging_utils import append_report
from utils.model_utils import get_gpu_info

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
ADAPTER_PATH = PROJECT_ROOT / "outputs" / "checkpoints"
VALID_MODES = {"--think", "--nothink"}


def get_inference_dtype() -> torch.dtype | str:
    if not torch.cuda.is_available():
        return "auto"
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def parse_args(argv: list[str]) -> tuple[str, str]:
    if len(argv) < 3:
        raise SystemExit(
            'Usage:\npython chat_lora.py --think "Hello"\npython chat_lora.py --nothink "Hello"'
        )

    mode = argv[1]
    if mode not in VALID_MODES:
        raise SystemExit("Invalid mode. Use --think or --nothink")

    return mode, argv[2]

def main() -> None:
    mode, original_prompt = parse_args(sys.argv)
    prompt = f"/no_think\n{original_prompt}" if mode == "--nothink" else original_prompt

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    startup_start = time.perf_counter()
    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer_load_time = time.perf_counter() - tokenizer_start

    base_model_start = time.perf_counter()
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=get_inference_dtype(),
        attn_implementation="sdpa",
        device_map="auto",
    )
    base_model_load_time = time.perf_counter() - base_model_start

    adapter_start = time.perf_counter()
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()
    adapter_load_time = time.perf_counter() - adapter_start

    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    torch.manual_seed(42)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
    generation_time = time.perf_counter() - generation_start

    generated = outputs[0][prompt_tokens:]
    generated_tokens = len(generated)
    response = tokenizer.decode(generated, skip_special_tokens=True)
    raw_response = tokenizer.decode(generated, skip_special_tokens=False)

    tokens_per_second = generated_tokens / generation_time if generation_time else 0.0
    total_runtime = time.perf_counter() - startup_start
    gpu_info = get_gpu_info()

    print("\n" + "=" * 60)
    print("BEACON LORA MODEL TEST")
    print("=" * 60)
    print(f"Time:      {timestamp_str}")
    print(f"Device:    {model.device}")
    print(f"GPU:       {gpu_info['gpu_name']}")
    print("\nPrompt:")
    print(original_prompt)
    print("\nResponse:")
    print(response)
    print("\n" + "-" * 60)
    print(f"Generated: {generated_tokens} tokens | {tokens_per_second:.2f} tok/s | {generation_time:.2f}s")
    print("=" * 60)

    append_report(
        f"""
================================================================================
Timestamp: {timestamp_str}
Mode: {mode}
Model: {MODEL_PATH}
Model Type: LoRA

Prompt:
{original_prompt}

Chat Template:
{chat_text}

Raw Response:
{raw_response}

Response:
{response}

METRICS

Tokenizer Load: {tokenizer_load_time:.2f}s
Base Model Load: {base_model_load_time:.2f}s
Adapter Load: {adapter_load_time:.2f}s
Generation Time: {generation_time:.2f}s
Total Runtime: {total_runtime:.2f}s

Prompt Tokens: {prompt_tokens}
Generated Tokens: {generated_tokens}

Tokens/sec: {tokens_per_second:.2f}

GPU: {gpu_info['gpu_name']}
VRAM Allocated: {gpu_info['vram_allocated']:.2f} GB
VRAM Reserved: {gpu_info['vram_reserved']:.2f} GB

"""
    )


if __name__ == "__main__":
    main()
