"""Comprehensive Evaluation Benchmark for Burmese Language Models (LoRA & Base).

Supports:
- Evaluating trained LoRA model
- Evaluating original Base model
- Side-by-side comparison mode (--compare)
- Generating detailed Markdown and Text benchmark reports
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
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
DEFAULT_PROMPTS_FILE = PROJECT_ROOT / "prompts" / "evaluation_prompts.json"
RESULTS_DIR = PROJECT_ROOT / "outputs" / "evaluations" / "results"
DEFAULT_SYSTEM_PROMPT = "သင်သည် အကူအညီပေးသော မြန်မာ AI လက်ထောက်တစ်ဦး ဖြစ်ပါသည်။"


def get_inference_dtype() -> torch.dtype | str:
    if not torch.cuda.is_available():
        return "auto"
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_prompts(prompts_path: Path, max_samples: int | None = None) -> list[dict]:
    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found at: {prompts_path}")

    prompts = []
    if prompts_path.suffix == ".jsonl":
        with prompts_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue

                if "messages" in data and isinstance(data["messages"], list):
                    user_msg = next((m.get("content", "") for m in data["messages"] if m.get("role") == "user"), "")
                    ref_msg = next((m.get("content", "") for m in data["messages"] if m.get("role") == "assistant"), "")
                    if user_msg:
                        prompts.append({
                            "category": "Test Set",
                            "prompt": user_msg,
                            "reference": ref_msg,
                        })
                else:
                    u = data.get("prompt") or data.get("instruction") or data.get("question") or ""
                    r = data.get("response") or data.get("output") or data.get("answer") or ""
                    if u:
                        prompts.append({
                            "category": "Test Set",
                            "prompt": str(u),
                            "reference": str(r),
                        })

                if max_samples and len(prompts) >= max_samples:
                    break
        return prompts

    with prompts_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if isinstance(item, dict):
            prompts.append({
                "category": item.get("category", "General"),
                "prompt": item.get("prompt", ""),
            })
        elif isinstance(item, str):
            prompts.append({
                "category": "General",
                "prompt": item,
            })
        if max_samples and len(prompts) >= max_samples:
            break
    return prompts


def generate_response(
    model: AutoModelForCausalLM | PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    think_mode: bool = False,
) -> tuple[str, int, float, float]:
    """Generate response and return (text, token_count, elapsed_sec, tok_per_sec)."""
    full_prompt = prompt if think_mode else f"/no_think\n{prompt}"
    messages = [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
    prompt_tokens = inputs.input_ids.shape[1]

    torch.manual_seed(42)
    start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(**inputs, **GENERATION_CONFIG)
    elapsed = time.perf_counter() - start_time

    generated_ids = outputs[0][prompt_tokens:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    tok_count = len(generated_ids)
    tps = tok_count / elapsed if elapsed > 0 else 0.0

    return response, tok_count, elapsed, tps


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Burmese AI Model on Benchmark Prompts")
    parser.add_argument("--base", action="store_true", help="Evaluate Base model only")
    parser.add_argument("--compare", action="store_true", help="Run side-by-side comparison of Base vs LoRA")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode (omit /no_think)")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL_PATH), help="Base model directory")
    parser.add_argument("--lora_path", type=str, default=str(DEFAULT_LORA_PATH), help="LoRA checkpoint directory")
    parser.add_argument("--prompts_file", type=str, default=str(DEFAULT_PROMPTS_FILE), help="Path to prompts JSON or JSONL")
    parser.add_argument("--test_set", action="store_true", help="Evaluate on datasets/clean/test/test_combined.jsonl")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of test samples to evaluate")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    prompts_path = PROJECT_ROOT / "datasets" / "clean" / "test" / "test_combined.jsonl" if args.test_set else Path(args.prompts_file)
    prompts = load_prompts(prompts_path, max_samples=args.max_samples)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    timestamp = datetime.now()
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    file_timestamp = timestamp.strftime("%Y%m%d_%H%M%S")
    gpu = get_gpu_info()

    print("=" * 80)
    print("BEACON MYANMAR AI - BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"Timestamp:    {timestamp_str}")
    print(f"GPU Device:   {gpu['gpu_name']} (Allocated: {gpu['vram_allocated']:.2f} GB)")
    print(f"Total Prompts:{len(prompts)} questions")
    print(f"Mode:         {'Side-by-side Comparison' if args.compare else ('Base Model Only' if args.base else 'LoRA Model')}")
    print(f"Thinking:     {'Enabled' if args.think else 'Disabled (/no_think)'}")
    print("=" * 80)

    print(f"\nLoading Tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading Base Model ({get_inference_dtype()})...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=get_inference_dtype(),
        attn_implementation="sdpa",
        device_map="auto" if torch.cuda.is_available() else None,
    )
    base_model.eval()

    lora_model = None
    if not args.base or args.compare:
        lora_path = Path(args.lora_path)
        if not lora_path.exists() or not (lora_path / "adapter_model.safetensors").exists():
            print(f"WARNING: LoRA adapter not found at {lora_path}. Falling back to Base model evaluation.")
        else:
            print(f"Loading LoRA Adapter from {lora_path}...")
            lora_model = PeftModel.from_pretrained(base_model, str(lora_path))
            lora_model.eval()

    eval_mode_name = "Side-by-Side (Base vs LoRA)" if (args.compare and lora_model) else ("Base Model" if args.base or not lora_model else "LoRA Model")

    # Evaluation loop
    results = []
    print("\n" + "=" * 80)
    print("STARTING BENCHMARK RUN")
    print("=" * 80 + "\n")

    current_category = None
    total_tokens = 0
    total_time = 0.0

    for idx, p in enumerate(prompts, 1):
        cat = p["category"]
        prompt_text = p["prompt"]

        if cat != current_category:
            current_category = cat
            print(f"\n▶ [{cat.upper()}]")
            print("-" * 70)

        print(f"\n[Test {idx}/{len(prompts)}] Q: {prompt_text}")

        item_result = {
            "index": idx,
            "category": cat,
            "prompt": prompt_text,
        }

        if args.compare and lora_model:
            # Generate Base
            base_resp, b_toks, b_time, b_tps = generate_response(base_model, tokenizer, prompt_text, args.think)
            # Generate LoRA
            lora_resp, l_toks, l_time, l_tps = generate_response(lora_model, tokenizer, prompt_text, args.think)

            print(f"  [Base]: {base_resp[:90]}... ({b_toks} tok | {b_tps:.1f} tps)")
            print(f"  [LoRA]: {lora_resp[:90]}... ({l_toks} tok | {l_tps:.1f} tps)")

            item_result["base_response"] = base_resp
            item_result["lora_response"] = lora_resp
            item_result["base_metrics"] = {"tokens": b_toks, "time": b_time, "tps": b_tps}
            item_result["lora_metrics"] = {"tokens": l_toks, "time": l_time, "tps": l_tps}
            total_tokens += l_toks
            total_time += l_time

        else:
            active_model = lora_model if (lora_model and not args.base) else base_model
            model_tag = "LoRA" if (lora_model and not args.base) else "Base"

            resp, toks, elapsed, tps = generate_response(active_model, tokenizer, prompt_text, args.think)
            print(f"  [{model_tag}]: {resp}")
            print(f"  └─ {toks} tokens | {elapsed:.2f}s | {tps:.1f} tok/s")

            item_result["response"] = resp
            item_result["metrics"] = {"tokens": toks, "time": elapsed, "tps": tps}
            total_tokens += toks
            total_time += elapsed

        results.append(item_result)

    # Compile Reports
    avg_tps = total_tokens / total_time if total_time > 0 else 0.0

    # 1. Generate Markdown Report
    md_lines = [
        "# BEACON Myanmar AI Evaluation Report",
        f"- **Date & Time:** {timestamp_str}",
        f"- **Evaluation Mode:** {eval_mode_name}",
        f"- **GPU Hardware:** {gpu['gpu_name']}",
        f"- **Total Benchmark Questions:** {len(prompts)}",
        f"- **Total Generated Tokens:** {total_tokens:,}",
        f"- **Average Throughput:** {avg_tps:.2f} tokens/sec",
        "\n---\n",
        "## Benchmark Results by Question\n",
    ]

    for r in results:
        md_lines.append(f"### Question {r['index']}: {r['prompt']}")
        md_lines.append(f"**Category:** `{r['category']}`\n")

        if args.compare and lora_model:
            bm = r["base_metrics"]
            lm = r["lora_metrics"]
            md_lines.append("#### 🔹 Base Model Response:")
            md_lines.append(f"> {r['base_response']}\n")
            md_lines.append(f"*Metrics:* `{bm['tokens']} tokens | {bm['time']:.2f}s | {bm['tps']:.1f} tok/s`\n")

            md_lines.append("#### 🔸 LoRA Adapted Model Response:")
            md_lines.append(f"> {r['lora_response']}\n")
            md_lines.append(f"*Metrics:* `{lm['tokens']} tokens | {lm['time']:.2f}s | {lm['tps']:.1f} tok/s`\n")
        else:
            m = r["metrics"]
            md_lines.append(f"> {r['response']}\n")
            md_lines.append(f"*Performance:* `{m['tokens']} tokens | {m['time']:.2f}s | {m['tps']:.1f} tok/s`\n")

        md_lines.append("---\n")

    md_file = RESULTS_DIR / f"eval_report_{file_timestamp}.md"
    md_file.write_text("\n".join(md_lines), encoding="utf-8")

    # 2. Generate Text Report
    txt_lines = [
        "=" * 80,
        "BEACON MYANMAR AI - EVALUATION REPORT",
        "=" * 80,
        f"Timestamp:   {timestamp_str}",
        f"Mode:        {eval_mode_name}",
        f"GPU:         {gpu['gpu_name']}",
        f"Prompts:     {len(prompts)}",
        f"Avg TPS:     {avg_tps:.2f} tok/s",
        "=" * 80 + "\n",
    ]

    for r in results:
        txt_lines.append(f"[TEST {r['index']}] Category: {r['category']}")
        txt_lines.append(f"Prompt: {r['prompt']}")
        if args.compare and lora_model:
            txt_lines.append(f"Base:   {r['base_response']}")
            txt_lines.append(f"LoRA:   {r['lora_response']}")
        else:
            txt_lines.append(f"Answer: {r['response']}")
        txt_lines.append("-" * 80)

    txt_file = RESULTS_DIR / f"eval_report_{file_timestamp}.txt"
    txt_file.write_text("\n".join(txt_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION COMPLETE")
    print(f"  Total Prompts Evaluated: {len(prompts)}")
    print(f"  Total Tokens Generated:  {total_tokens:,}")
    print(f"  Average Speed:           {avg_tps:.2f} tokens/sec")
    print(f"  Markdown Report Saved:   {md_file.relative_to(PROJECT_ROOT)}")
    print(f"  Text Report Saved:       {txt_file.relative_to(PROJECT_ROOT)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
