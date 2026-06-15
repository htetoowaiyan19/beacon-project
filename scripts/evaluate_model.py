# train ထားသော AI အား evaluation ဖြင့် prompts အများအပြားသုံးခါ စမ်းသပ်ရာတွင် သုံးသော script ဖြစ်သည်။

import json
import time

from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from utils.generation_utils import GENERATION_CONFIG
from utils.model_utils import get_gpu_info

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "../models/qwen3-4b"
PROMPTS_FILE = ("../prompts/evaluation_prompts.json") # ဤနေရာတွင် evaluation အတွက် အသုံးပြုမည့် prompts များပြင်နိုင်သည်။

# ============================================================
# TIMESTAMP
# ============================================================

timestamp = datetime.now()
timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
report_date = timestamp.strftime("%Y%m%d")

# ============================================================
# LOAD PROMPTS
# ============================================================

with open(
    PROMPTS_FILE,
    "r",
    encoding="utf-8"
) as file:
    prompts = json.load(file)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)

gpu_info = get_gpu_info()

# ============================================================
# REPORT SETUP
# ============================================================

report_lines = []
report_lines.append("=" * 80)
report_lines.append("BEACON EVALUATION REPORT")
report_lines.append("=" * 80)
report_lines.append(f"Timestamp: {timestamp_str}")
report_lines.append(f"Model: {MODEL_PATH}")
report_lines.append(f"GPU: {gpu_info['gpu_name']}")
report_lines.append("")

# ============================================================
# STATISTICS
# ============================================================

total_generation_time = 0
total_generated_tokens = 0
total_tokens_per_second = 0

prompt_count = len(prompts)

# ============================================================
# RUN TESTS
# ============================================================

print()
print("=" * 60)
print("BEACON EVALUATION")
print("=" * 60)
print()

for index, item in enumerate(prompts, start=1):
    if isinstance(item, dict):
        prompt = item["prompt"]

        category = item.get("category", "uncategorized")

    else:

        prompt = item
        category = "uncategorized"

    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        text,
        return_tensors="pt"
    ).to(model.device)

    prompt_tokens = (inputs.input_ids.shape[1])
    start_time = time.perf_counter()
    outputs = model.generate(**inputs, **GENERATION_CONFIG)

    generation_time = (time.perf_counter() -start_time)
    
    generated = outputs[0][
        prompt_tokens:
    ]

    generated_tokens = len(generated)
    tokens_per_second = generated_tokens / generation_time

    response = tokenizer.decode(
        generated,
        skip_special_tokens=True
    )

    # --------------------------
    # Statistics
    # --------------------------

    total_generation_time += generation_time
    total_generated_tokens += generated_tokens
    total_tokens_per_second += tokens_per_second

    # --------------------------
    # Terminal
    # --------------------------

    print(
        f"[{index}/{prompt_count}] "
        f"{prompt}"
    )

    print(
        f"✓ {generated_tokens} tokens | "
        f"{generation_time:.2f}s"
    )

    print()

    # --------------------------
    # Report
    # --------------------------

    report_lines.extend([
        "=" * 80,
        f"TEST #{index}",
        "=" * 80,
        "",
        f"Category: {category}",
        "",
        "Prompt:",
        prompt,
        "",
        "Response:",
        response,
        "",
        f"Prompt Tokens: {prompt_tokens}",
        f"Generated Tokens: {generated_tokens}",
        f"Generation Time: {generation_time:.2f}s",
        f"Tokens/sec: {tokens_per_second:.2f}",
        ""
    ])

# ============================================================
# SUMMARY
# ============================================================

average_time = total_generation_time / prompt_count
average_tokens_per_second = total_tokens_per_second / prompt_count

report_lines.extend([
    "=" * 80,
    "SUMMARY",
    "=" * 80,
    "",
    f"Prompts Tested: {prompt_count}",
    f"Total Generated Tokens: {total_generated_tokens}",
    f"Total Generation Time: {total_generation_time:.2f}s",
    f"Average Generation Time: {average_time:.2f}s",
    f"Average Tokens/sec: {average_tokens_per_second:.2f}",
    ""
])

# ============================================================
# SAVE REPORT
# ============================================================

report_path = (
    f"../outputs/evaluations/results/"
    f"e{report_date}.txt"
)

with open(
    report_path,
    "a",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report_lines)
    )

    file.write("\n\n")

# ============================================================
# TERMINAL SUMMARY
# ============================================================

print("=" * 60)
print("Completed")
print("=" * 60)

print(
    f"Prompts Tested : "
    f"{prompt_count}"
)

print(
    f"Total Tokens   : "
    f"{total_generated_tokens}"
)

print(
    f"Average Speed  : "
    f"{average_tokens_per_second:.2f} tok/s"
)

print(
    f"Total Time     : "
    f"{total_generation_time:.2f}s"
)

print(
    f"Report Saved   : "
    f"{report_path}"
)

print("=" * 60)