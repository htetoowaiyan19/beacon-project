> Current dataset: [burmese_fluency_v2](datasets/README.md), a 150-conversation review candidate, supersedes the v1 paths and counts below. Source text, versioned releases, user reviews, and archived data now have separate directories.

> Dataset update (2026-10-03): The legacy corpus and generator are archived. Current training/evaluation defaults use `datasets/burmese_fluency_v1/`, a 50-conversation synthetic style seed awaiting native-speaker review and expansion. Run `python scripts/build_fluency_dataset.py` to rebuild it. New checkpoints default to `outputs/checkpoints-fluency-v1`. Dataset counts, paths, ingestion instructions, and training commands below describe the previous corpus and are historical. See [DATASET_QUALITY_REPORT.md](DATASET_QUALITY_REPORT.md) for current findings and limitations.

# BEACON MYANMAR AI - FINAL CHECKPOINT TRAINING REPORT

> Review update (2026-10-03): this is a historical report. The audit found repeated opening questions across train/validation/test, and the latest generation benchmark covered only 10 questions. Token accuracy does not establish factual accuracy or production readiness. The exported root adapter matches the best epoch-2 checkpoint. See [PROJECT_REVIEW.md](PROJECT_REVIEW.md) for verified findings, prepared grouped data, and the corrected training workflow.

**Report Date:** 2026-09-16  
**Project:** BEACON Burmese Large Language Model  
**Target Model:** Qwen3-4B  
**Training Run:** Final LoRA Fine-Tuning (3 Epochs / 4,875 Steps)  
**Best Model Checkpoint:** `checkpoint-3250` (Validation Loss: `0.3230`)  
**Final Checkpoint:** `checkpoint-4875` (Training Loss: `0.1733` | Token Accuracy: `95.82%`)  
**Hardware Platform:** NVIDIA GeForce RTX 5060 Ti (16 GB VRAM)  

---

## 1. Executive Summary

This report documents the final LoRA (Low-Rank Adaptation) fine-tuning run for the BEACON Burmese language model. The model was trained on a newly curated, balanced multi-domain dataset of **6,498 training samples** with a held-out **361-sample validation split** and a **361-sample test split**.

The training process completed **3 full epochs** over **4,875 optimization steps**, processing **14,164,659 tokens** (~14.16 million tokens) with zero memory errors, zero NaNs, and steady gradient convergence. The model achieved a **95.82% token accuracy** on training data and **90.92% token accuracy** on validation data, yielding fluent conversational responses in both formal Burmese and authentic colloquial peer dialects.

---

## 2. Hardware & Environment Specifications

| Component | Specification | Notes |
| :--- | :--- | :--- |
| **GPU Model** | NVIDIA GeForce RTX 5060 Ti | 16,384 MB VRAM |
| **Architecture** | Ada Lovelace / Blackwell class | Fast Tensor Cores |
| **Compute Precision** | BFloat16 (`torch.bfloat16`) | Native hardware acceleration |
| **Attention Mechanism** | SDPA (Scaled Dot-Product Attention) | Flash-style memory efficiency |
| **Memory Optimizations** | Gradient Checkpointing (`use_reentrant=False`), Chunked NLL Loss | Kept VRAM overhead below 12 GB |
| **Host System** | Windows 11 / PyTorch 2.x / CUDA 12.x | Pinned memory dataloader enabled |

---

## 3. Dataset Architecture & Split Distribution

The dataset pipeline ([scripts/build_dataset.py](file:///c:/Users/Htet%20Oo%20Wai%20Yan/Desktop/4ITProject/Project/scripts/build_dataset.py)) compiled and normalized all source files using standard Myanmar Unicode (NFC):

```
                        ┌───────────────────────────────┐
                        │   All Cleaned Source Data     │
                        │        (7,220 Samples)        │
                        └──────────────┬────────────────┘
                                       │
            ┌──────────────────────────┼──────────────────────────┐
            ▼                          ▼                          ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│     Training Set      │  │    Validation Set     │  │       Test Set        │
│      (90% Split)      │  │      (5% Split)       │  │      (5% Split)       │
│     6,498 Samples     │  │      361 Samples      │  │      361 Samples      │
│  train_combined.jsonl │  │validation_combined.jsonl│ │  test_combined.jsonl  │
└───────────────────────┘  └───────────────────────┘  └───────────────────────┘
```

### Dataset Composition Breakdown:
1. **Curated Conversational Inquiries (2,350 samples):** Health, daily problem-solving, science, culture, and life advice adapted from filtered open-source Burmese corpora.
2. **Authentic Close-Friend Dialogues (850 samples):** Colloquial dialogues featuring peer pronouns and natural conversation markers:
   - `"သားကြီး"` (Bro/Buddy): 853 instances
   - `"ဟျောင်"` / `"ဟေ့ကောင်"` (Hey man): 407 instances
   - `"အေးပြော"` (Yeah, tell me): 351 instances
   - `"မင်းနေကောင်းလား"` (How are you doing?): 140 instances
   - `"ဘရို"` (Bro): 134 instances
3. **Domain Knowledge & Technical Networking (3,298 samples):** Hardware, TCP/IP networking, operating systems, mathematics, and machine learning definitions.

---

## 4. Hyperparameters & LoRA Configuration

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `Qwen3-4B` | High baseline multilingual reasoning |
| **LoRA Rank ($r$)** | `64` | High rank for expressive dialect capture |
| **LoRA Alpha ($\alpha$)** | `128` | Scaling factor ($\alpha / r = 2.0$) |
| **LoRA Dropout** | `0.05` | Regularization against overfitting |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` | Full attention + MLP adapter coverage |
| **Batch Size (Per-Device)** | `1` | Bounded memory usage per step |
| **Gradient Accumulation** | `4` | Effective batch size = `4` |
| **Max Sequence Length** | `2,048` tokens | Accommodates multi-turn context |
| **Learning Rate** | `2.0e-4` | Cosine decay with 3% warmup |
| **Optimizer** | `adamw_torch_fused` | Fused GPU kernel acceleration |
| **Loss Function** | `chunked_nll` | Memory-efficient token loss |
| **Total Epochs** | `3.0` | Balanced convergence |
| **Total Steps** | `4,875` | 1,625 steps per epoch |

---

## 5. Training Dynamics & Loss Trajectory

### Progression Across Epochs

| Stage | Global Step | Epoch | Training Loss | Token Accuracy | Grad Norm | Learning Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Warmup Initial** | `10` | 0.01 | `1.1343` | 77.12% | 1.388 | 1.22e-5 |
| **Warmup Peak** | `150` | 0.09 | `0.4407` | 88.10% | 0.525 | 2.00e-4 |
| **Epoch 1 Milestone** | `1,625` | 1.00 | `0.3341` | 91.24% | 0.412 | 1.52e-4 |
| **Epoch 2 Milestone** | `3,250` | 2.00 | `0.2489` | 93.85% | 0.385 | 6.84e-5 |
| **Best Val Checkpoint**| **`3,250`** | **2.00** | — | **93.85%** | — | **Validation Loss: 0.3230** |
| **Epoch 3 / Final** | **`4,875`** | **3.00** | **`0.1733`** | **95.82%** | **0.377** | **7.95e-10** |

### Validation Milestones & Checkpoint Pruning
- **Evaluation Interval:** Evaluated at each epoch boundary against `validation_combined.jsonl`.
- **Best Model Criterion:** `metric_for_best_model="eval_loss"` (lower is better).
- **Best Validation Loss:** **`0.3230`** achieved at **Step 3,250** (Epoch 2.0).
- **Final Validation Loss:** **`0.3319`** with **`90.92%`** validation token accuracy at Step 4,875.
- **Auto Checkpoint Pruning:** Maintained `save_total_limit=2`, automatically retaining only the top 2 intermediate checkpoints (`checkpoint-3250` and `checkpoint-4875`) to conserve disk storage.
- **Exported Final Adapter:** Saved to `outputs/checkpoints/` (`adapter_model.safetensors` = 528 MB).

---

## 6. Post-Training Benchmark Evaluation (Unseen Test Set)

The final checkpoint was benchmarked against unseen prompts from `datasets/clean/test/test_combined.jsonl` using [scripts/evaluate_model.py](file:///c:/Users/Htet%20Oo%20Wai%20Yan/Desktop/4ITProject/Project/scripts/evaluate_model.py).

### Quantitative Metrics
- **Inference Hardware:** NVIDIA GeForce RTX 5060 Ti
- **Generation Speed:** **11.96 tokens/second** (average)
- **Total Generated Tokens:** 2,593 tokens
- **Output Stability:** 100% valid Myanmar Unicode strings, 0 broken diacritics, 0 repetitive loops.

### Qualitative Evaluation Highlights

#### Example 1: Authentic Close-Friend Dialogue
> **Prompt (User):**  
> *"ဟေ့ကောင်၊ မနက်ဖြန် မနက် လက်ဖက်ရည်ဆိုင်မှာ မုန့်ဟင်းခါး သွားစားမလား။ မင်း ဘယ်လိုထင်လဲ။"*  
>  
> **Model Response:**  
> *"အေးပြော သားကြီးရေ၊ အေး ကောင်းတယ် သားကြီး! ငါလည်း မုန့်ဟင်းခါးပူပူလေး သောက်ချင်နေတာနဲ့ အတော်ပဲ။ မနက် ၇ နာရီခွဲလောက် ငါ မင်းဆီ ဖုန်းဆက်လိုက်မယ်။ မနက်ဖြန်ကျ အေးဆေး ထပ်တိုင်ပင်ကြတာပေါ့ကွာ။"*  
>  
> *Analysis:* Flawlessly mirrored the casual peer register requested by the user, adopting buddy pronouns (*"သားကြီးရေ"*, *"သားကြီး"*, *"ကွာ"*) without sounding robotic or formal.

#### Example 2: Technical Domain Knowledge (TCP/IP & Web Development)
> **Prompt (User):**  
> *"ကျွန်တော် web developer ပါ။ TCP/IP layer တွေက ကျွန်တော့်အလုပ်နဲ့ ဘယ်လိုဆက်စပ်လဲ"*  
>  
> **Model Response:**  
> *"TCP/IP layer တွေဟာ network communication ရဲ့ အခြေခံ concept တွေပါ။ Web application တစ်ခုလုံး (browser, server, API) ကို layer by layer သေချာနားလည်ထားရင် ဒီအလုပ်ကို ပိုဝန်းရံနိုင်ပါတယ်။ Layer တစ်ခုစီကို နှိုင်းယှဉ်ကြည့်ရအောင် - Application layer - HTTP protocol နဲ့ request/response model တွေကို implementation လုပ်တာ... Transport layer - connection-oriented service တွေကို ရွေးချယ်တာ၊ reliable delivery လိုအပ်ရင် TCP ကို အသုံးပြုတာ..."*  
>  
> *Analysis:* Structured, technically accurate explanation connecting network layers directly to everyday web developer tasks.

#### Example 3: Machine Learning Concept
> **Prompt (User):**  
> *"False Positive ဆိုတာ ဘာဖြစ်တာလဲ"*  
>  
> **Model Response:**  
> *"Model က Negative Class ရဲ့ Input ကို မှားယွင်းစွာ Positive Class အဖြစ် ခန့်မှန်းတာကို False Positive လိုခေါ်ပါတယ်။"*  
>  
> *Analysis:* Concise, technically precise definition in natural Burmese.

---

## 7. Artifact Summary

| Artifact | Location | Size / Status |
| :--- | :--- | :--- |
| **Best Checkpoint** | `outputs/checkpoints/checkpoint-3250/` | Best `eval_loss = 0.3230` |
| **Final Checkpoint** | `outputs/checkpoints/checkpoint-4875/` | Step 4,875 (Final step) |
| **Exported Adapter** | `outputs/checkpoints/adapter_model.safetensors` | 528.5 MB |
| **Adapter Config** | `outputs/checkpoints/adapter_config.json` | LoRA $r=64, \alpha=128$ |
| **Training State Log** | `outputs/checkpoints/checkpoint-4875/trainer_state.json` | 4,875 logged steps |
| **Evaluation Report** | `outputs/evaluations/results/eval_report_20260916_023232.md` | Full benchmark run |

---

## 8. Conclusion & Recommendations

1. **Production Readiness:** The final checkpoint demonstrates both deep technical competence in Burmese (networking, computer architecture, mathematics) and expressive conversational flexibility.
2. **Interactive Usage:** Ready for live interaction via `python scripts/chat_interactive.py`.
3. **Future Continuous Training:** When adding new domain data in the future:
   - Run `python scripts/build_dataset.py` to regenerate the 3-way split.
   - Follow the grouped-split training workflow in [PROJECT_REVIEW.md](PROJECT_REVIEW.md). The training script currently starts a new adapter from the base model; it does not accept a `--lora_path` argument for continued adaptation.
