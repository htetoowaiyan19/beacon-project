"""Interactive CLI chat with the Burmese LoRA or Base model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from utils.generation_utils import GENERATION_CONFIG
from utils.model_utils import get_gpu_info

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "qwen3-4b"
DEFAULT_LORA_PATH = PROJECT_ROOT / "outputs" / "checkpoints"
DEFAULT_SYSTEM_PROMPT = "သင်သည် အကူအညီပေးသော မြန်မာ AI လက်ထောက်တစ်ဦး ဖြစ်ပါသည်။"


def get_inference_dtype() -> torch.dtype | str:
    if not torch.cuda.is_available():
        return "auto"
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Burmese Model Chat")
    parser.add_argument("--base", action="store_true", help="Chat with base model instead of LoRA")
    parser.add_argument("--think", action="store_true", help="Enable thinking tags if model supports them")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Base model directory")
    parser.add_argument("--lora_path", type=str, default=str(DEFAULT_LORA_PATH), help="LoRA checkpoint directory")
    args = parser.parse_args()

    print("=" * 70)
    print("BEACON MYANMAR AI - INTERACTIVE CHAT")
    print("=" * 70)
    print(f"Mode:        {'Base Model' if args.base else 'LoRA Adapted Model'}")
    print(f"Thinking:    {'Enabled' if args.think else 'Disabled (/no_think)'}")
    print("Loading models... Please wait...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_inference_dtype(),
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
    )

    if not args.base and Path(args.lora_path).exists():
        print(f"Loading LoRA adapter from {args.lora_path}...")
        model = PeftModel.from_pretrained(base_model, args.lora_path)
    else:
        model = base_model

    model.eval()
    gpu = get_gpu_info()
    print(f"Device:      {gpu['gpu_name']}")
    print("-" * 70)
    print("Commands: 'exit' / 'quit' to exit, 'clear' to reset chat history.")
    print("=" * 70 + "\n")

    history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]

    while True:
        try:
            user_input = input("User (မြန်မာ/English) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting chat. Bye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Exiting chat. မင်္ဂလာရှိသော နေ့လေးဖြစ်ပါစေ!")
            break

        if user_input.lower() in {"clear", "reset"}:
            history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
            print("Chat history reset.\n")
            continue

        prompt = user_input if args.think else f"/no_think\n{user_input}"
        history.append({"role": "user", "content": prompt})

        chat_text = tokenizer.apply_chat_template(
            history,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        prompt_tokens = inputs.input_ids.shape[1]

        start_time = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(**inputs, **GENERATION_CONFIG)
        elapsed = time.perf_counter() - start_time

        generated_ids = outputs[0][prompt_tokens:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        tps = len(generated_ids) / elapsed if elapsed > 0 else 0

        print(f"\nAssistant > {response}")
        print(f"[{len(generated_ids)} tokens | {elapsed:.2f}s | {tps:.1f} tok/s]\n")

        # Keep natural history without prompt prefix
        history[-1]["content"] = user_input
        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
