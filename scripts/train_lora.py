# AI train ရန် အသုံးပြုတဲ့ Script ဖြစ်ပါတယ်။

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from peft import get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer

# ဆက်လက် train နေသော AI Model: Qwen3 4B
# နည်းပညာ: LoRA (Low-Rank Adaptation)

MODEL_PATH = "../models/qwen3-4b" # မိမိ train မည့် model ရဲ့ path

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def format_chat(example):

    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False
        )
    }

dataset = load_dataset(
    "json",
    data_files="../datasets/clean/train/data_clean.jsonl" # မိမိ train လိုသော dataset ရဲ့ path
)

dataset = dataset.map(format_chat)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ]
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto"
)

model = get_peft_model(
    model,
    lora_config
)

model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="../outputs/checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    logging_steps=1,
    save_strategy="epoch",
    learning_rate=2e-4,
    fp16=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    args=training_args,
    processing_class=tokenizer
)

trainer.train()

model.save_pretrained(
    "../outputs/checkpoints"
)