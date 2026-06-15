# train ထားသော AI အား evaluation ဖြင့် prompts အများအပြားသုံးခါ စမ်းသပ်ရာတွင် သုံးသော script ဖြစ်သည်။

import sys, json, time
from pathlib import Path
from datetime import datetime
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from utils.generation_utils import GENERATION_CONFIG
from utils.model_utils import get_gpu_info

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "../models/qwen3-4b"
LORA_PATH = "../outputs/checkpoints"
PROMPTS_FILE = "../prompts/evaluation_prompts.json" # Evaluate လုပ်မည့် prompts များရှိသည့် json file

# ============================================================
# ARGUMENTS
# ============================================================

if len(sys.argv) < 2: print("Usage: python evaluate_model.py --think|--nothink"); sys.exit(1)
mode = sys.argv[1]
if mode not in ["--think", "--nothink"]: print("Invalid mode"); sys.exit(1)
prefix = "/no_think\n" if mode == "--nothink" else ""

# ============================================================
# TIME + PROMPTS
# ============================================================

now = datetime.now()
ts, date = now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y%m%d")

prompts = json.load(open(PROMPTS_FILE, "r", encoding="utf-8"))

Path("../outputs/evaluations/results").mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD MODEL (BASE + LORA)
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")

print("Loading LoRA...")
model = PeftModel.from_pretrained(base_model, LORA_PATH)

gpu = get_gpu_info()

# ============================================================
# REPORT
# ============================================================

report = []
report.append("="*80)
report.append("BEACON EVALUATION REPORT")
report.append("="*80)
report.append(f"Time: {ts}")
report.append(f"Model: {MODEL_PATH}")
report.append(f"Mode: {mode}")
report.append(f"GPU: {gpu['gpu_name']}\n")

total_t, total_tok = 0, 0

# ============================================================
# LOOP
# ============================================================

print("\n=== EVALUATION START ===\n")

for i, item in enumerate(prompts, 1):

    prompt = prefix + (item["prompt"] if isinstance(item, dict) else item)
    category = item.get("category", "uncategorized") if isinstance(item, dict) else "uncategorized"

    text = tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    p_tokens = inputs.input_ids.shape[1]

    start = time.perf_counter()
    out = model.generate(**inputs, **GENERATION_CONFIG)
    t = time.perf_counter() - start

    gen = out[0][p_tokens:]
    tok = len(gen)

    resp = tokenizer.decode(gen, skip_special_tokens=True)

    total_t += t
    total_tok += tok

    print(f"[{i}/{len(prompts)}] {tok} tok | {t:.2f}s")

    report.append("="*80)
    report.append(f"TEST {i} | {category}")
    report.append(f"Prompt: {prompt}")
    report.append(f"Response: {resp}")
    report.append(f"Tokens: {tok} | Time: {t:.2f}s | TPS: {tok/t:.2f}\n")

# ============================================================
# SUMMARY
# ============================================================

report.append("="*80)
report.append("SUMMARY")
report.append("="*80)
report.append(f"Prompts: {len(prompts)}")
report.append(f"Total Tokens: {total_tok}")
report.append(f"Total Time: {total_t:.2f}s")
report.append(f"Avg TPS: {total_tok/total_t:.2f}")

# ============================================================
# SAVE
# ============================================================

Path(f"../outputs/evaluations/results/e{date}.txt").write_text("\n".join(report), encoding="utf-8")

print("\n=== DONE ===")
print(f"Report saved: e{date}.txt")