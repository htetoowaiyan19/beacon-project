# ဒီ script က Plain Model ကို test လုပ်ရာတွင်သုံးပါတယ်၊ Model တစ်ခု down ပြီးပါက MODEL_PATH တွင်ပြင်၍ အသုံးပြုနိုင်ပါတယ်။

import sys
import time
import torch

from utils.model_utils import get_gpu_info
from utils.logging_utils import append_report
from utils.generation_utils import GENERATION_CONFIG

from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# ============================================================
# အသုံးပြုပုံ
# ============================================================
# CLI တွင်
# python chat_lora.py {think?} {str:prompt}
# think? = --think | --nothink
# --think ဆိုရင် AI က reasoning လုပ်ပြီးမှ response ဖြေမှာပါ
# --nothink ဆိုရင် resoning မလုပ်ပဲ တန်းဖြေပေးမှာပါ
# {str:prompt} ကတော့ AI ကိုမေးချင်တဲ့ prompt ကို double quote ("") ထဲမှာထည့်ရေးပါ
# ဥပမာ python chat_lora.py --nothink "မင်္ဂလာပါ"
# ============================================================

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "../models/qwen3-4b"

# ============================================================
# ARGUMENTS
# ============================================================

if len(sys.argv) < 3:
    print(
        "Usage:\n"
        "python chat_base.py --think \"Hello\"\n"
        "python chat_base.py --nothink \"Hello\""
    )
    sys.exit(1)

mode = sys.argv[1]
prompt = sys.argv[2]

VALID_MODES = [
    "--think",
    "--nothink"
]

if mode not in VALID_MODES:

    print(
        "Invalid mode.\n"
        "Use --think or --nothink"
    )

    sys.exit(1)

if mode == "--nothink":
    prompt = "/no_think\n" + prompt

# ============================================================
# TIMESTAMP
# ============================================================

script_start = time.perf_counter()

timestamp = datetime.now()

timestamp_str = timestamp.strftime(
    "%Y-%m-%d %H:%M:%S"
)

# ============================================================
# LOAD MODEL
# ============================================================

tokenizer_start = time.perf_counter()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

tokenizer_load_time = (
    time.perf_counter() -
    tokenizer_start
)

model_start = time.perf_counter()

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)

model_load_time = (
    time.perf_counter() - model_start
)

# ============================================================
# BUILD PROMPT
# ============================================================

messages = [
    {
        "role": "user",
        "content": prompt
    }
]

chat_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(
    chat_text,
    return_tensors="pt"
).to(model.device)

prompt_tokens = inputs.input_ids.shape[1]

# ============================================================
# GENERATE
# ============================================================

torch.manual_seed(42)

generation_start = time.perf_counter()

outputs = model.generate(
    **inputs,
    **GENERATION_CONFIG
)

generation_time = (
    time.perf_counter() -
    generation_start
)

generated = outputs[0][prompt_tokens:]

generated_tokens = len(generated)

response = tokenizer.decode(
    generated,
    skip_special_tokens=True
)

raw_response = tokenizer.decode(
    generated,
    skip_special_tokens=False
)

# ============================================================
# METRICS
# ============================================================

tokens_per_second = (
    generated_tokens /
    generation_time
)

total_runtime = (
    time.perf_counter() -
    script_start
)

# ============================================================
# TERMINAL OUTPUT
# ============================================================

gpu_info = get_gpu_info()

print("\n" + "=" * 60)
print("BEACON BASE MODEL TEST")
print("=" * 60)

print(f"Time:      {timestamp_str}")
print(f"Device:    {model.device}")
print(f"GPU:       {gpu_info['gpu_name']}")

print("\nPrompt:")
print(sys.argv[2])

print("\nResponse:")
print(response)

print("\n" + "-" * 60)
print(
    f"Generated: {generated_tokens} tokens | "
    f"{tokens_per_second:.2f} tok/s | "
    f"{generation_time:.2f}s"
)

print("=" * 60)

# ============================================================
# SAVE EVALUATION
# ============================================================

report = f"""
================================================================================
Timestamp: {timestamp_str}
Mode: {mode}
Model: {MODEL_PATH}
Model Type: Base

Prompt:
{sys.argv[2]}

Chat Template:
{chat_text}

Raw Response:
{raw_response}

Response:
{response}

METRICS

Tokenizer Load: {tokenizer_load_time:.2f}s
Model Load: {model_load_time:.2f}s
Generation Time: {generation_time:.2f}s
Total Runtime: {total_runtime:.2f}s

Prompt Tokens: {prompt_tokens}
Generated Tokens: {generated_tokens}

Tokens/sec: {tokens_per_second:.2f}

GPU: {gpu_info['gpu_name']}
VRAM Allocated: {gpu_info['vram_allocated']:.2f} GB
VRAM Reserved: {gpu_info['vram_reserved']:.2f} GB

"""

append_report(report)