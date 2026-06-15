# Model ကို မိမိ Computer ပေါ်သို့ Download လုပ်ရန် Script ဖြစ်ပါတယ်။ Hugging Face မှာရှိတဲ့ Open-Souce Model မှန်သမျှကို download နိုင်ပါတယ်။

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

model_name = "Qwen/Qwen3-4B" # Model Name
save_path = "../models/qwen3-4b" # Save လုပ်မဲ့ path

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"
)

print("Saving...")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("Done!")