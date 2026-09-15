# BEACON PROJECT - Burmese AI Assistant

An open initiative to build a fluent, culturally-aware Burmese (Myanmar Language) Large Language Model Assistant using Qwen3 and Parameter-Efficient Fine-Tuning (LoRA).

---

## Project Structure

```
beacon-project/
│
├── datasets/                   - Datasets repository
│   ├── clean/                  - Cleaned & validated datasets
│   │   ├── train/              - train_combined.jsonl (Ready for training)
│   │   └── validation/         - validation_combined.jsonl (Evaluation split)
│   ├── raw/                    - Raw collected JSONL files
│   └── trained/                - Archived training sets
│
├── models/                     - Base AI Models (e.g., Qwen3-4B)
│
├── outputs/
│   ├── checkpoints/            - LoRA weights & adapter configs
│   ├── evaluations/            - Automated benchmark reports
│   └── logs/                   - Runtime logs
│
├── prompts/
│   └── evaluation_prompts.json - Multi-domain Burmese benchmark prompts
│
├── scripts/
│   ├── build_dataset.py        - Clean, deduplicate & compile training splits
│   ├── train_lora.py           - Train LoRA adapter on RTX 5060Ti (16GB)
│   ├── chat_interactive.py     - Real-time interactive terminal chat
│   ├── chat_lora.py            - Test single prompt on LoRA model
│   ├── chat_base.py            - Test single prompt on Base model
│   ├── evaluate_model.py       - Run automated multi-domain evaluation benchmark
│   └── utils/                  - Text normalizer, generation & model utilities
│
├── requirements.txt
└── README.md
```

---

## Quickstart Guide

### 1. Build / Recompile the Combined Dataset
Merges all clean/raw datasets, normalizes Myanmar Unicode, strips synthetic noise, removes duplicates, and creates `train_combined.jsonl`:
```bash
python scripts/build_dataset.py
```

### 2. Train the LoRA Model (Ready in 1 Go)
Trains on the combined Burmese dataset with automatic VRAM optimization and checkpoint pruning (keeping only the best 2 checkpoints):
```bash
python scripts/train_lora.py
```
*Optional parameters:*
```bash
python scripts/train_lora.py --epochs 3 --batch_size 1 --grad_accum 4 --learning_rate 2e-4 --lora_r 64
```

### 3. Interactive Chat with the Model
Chat with your trained Burmese model in real-time:
```bash
# Chat with LoRA model
python scripts/chat_interactive.py

# Chat with Base model for comparison
python scripts/chat_interactive.py --base
```

### 4. Run Benchmark Evaluation
Evaluate model fluency across 15+ benchmark questions:
```bash
python scripts/evaluate_model.py --nothink
```

---

## Hardware Configuration (RTX 5060 Ti 16GB)

| Parameter | Setting | Reason |
| :--- | :--- | :--- |
| **Precision** | `bfloat16` (BF16) | Native acceleration on RTX 50-series |
| **Batch Size** | 1 (per device) | Conserves VRAM during peak attention |
| **Gradient Accumulation** | 4 steps | Effective batch size of 4 |
| **Max Sequence Length** | 2048 tokens | Fits complex multi-turn Burmese conversations |
| **LoRA Rank ($r$)** | 64 ($alpha = 128$) | High expressive capacity for Burmese language |
| **Loss Function** | `chunked_nll` | Prevents vocab matrix VRAM explosion |
