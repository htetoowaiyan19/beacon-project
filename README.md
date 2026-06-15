# BEACON PROJECT
As artificial intelligence continues to evolve year after year, we, as Myanmar, an underdeveloped country, aim to assist our citizens in effectively utilizing AI in our native language.

## Data Collection Team
မိမိ Collect လုပ်ထားသော data များကို [raw](https://github.com/htetoowaiyan19/beacon-project/tree/main/datasets/raw) ထဲမှ သက်ဆိုင်သော directory ထဲတွင် file တစ်ခုတည်းအဖြစ် စုပြီးထည့်ပေးပါ။ **မိမိ PC ထဲတွင် မဖျက်ပဲ Backup ထားပေးပါရန်။**

## Data Cleaning Team
[raw](https://github.com/htetoowaiyan19/beacon-project/tree/main/datasets/raw) ထဲမှ data များကို စစ်ပြီး [clean](https://github.com/htetoowaiyan19/beacon-project/tree/main/datasets/clean) ထဲသို့ပြောင်းထည့်ပေးပါ။ Clean ပြီးသော raw datasets များကို တစ်ခါတည်းဖျက်ပေးပါ။ သေချာပြီဆိုမှ repo ကဖျက်ပါ။ **မိမိ PC ထဲတွင် မဖျက်ပဲ Backup ထားပေးပါရန်။**

### Project Structure
```
beacon-project/
│
├── datasets/                   - Dataset နှင့်ဆိုင်သော file များ
│   ├── clean/                  - Clean ပြီးသား dataset များကိုသိမ်းရန်
│   │   ├── test/
│   │   ├── train/
│   │   └── validation/
│   ├── raw/                    - Collect ပြီးသာ dataset အသစ်များကိုသိမ်းရန်
│   │   ├── test/
│   │   ├── train/
│   │   └── validation/
│   └── trained/                - Train ပြီးသော dataset များ
│       ├── test/
│       ├── train/
│       └── validation/
│
├── models/                     - AI Model များထားသည့်နေရာ
│
├── notebooks/                  - Code စမ်းရန် Jupyter Notebooks များ
│   ├── test.ipynb
│   ├── tokenizer_test.ipynb
│   └── training_test.ipynb
│
├── outputs/
│   ├── checkpoints/            - LoRA train ထားသော checkpoints များ (gitignore)        
│   ├── evaluations/            - Evaluation record များ
│   │   ├── chat_results/       - chat_base နှင့် chat_lora မှရသော results များ
│   │   └── results/            - evaluation မှရသော results များ
│   └── logs/                   - Program log များ
│
├── prompts/                    - Evaluation အတွက် prompts များ
│
├── scripts/
│   ├── chat_base.py            - Train မလုပ်ထားသော model ကိုစမ်းရန်
│   ├── chat_lora.py            - Train ထားသော model ကိုစမ်းရန်
│   ├── download_model.py       - Model တစ်ခု download လုပ်ရန်
│   ├── evaluate_model.py       - Evaluation လုပ်ရန်
│   └── train_lora.py           - Model train ရန်
│
├── requirements.txt
│
└── README.md
```

### မိမိ PC သို့ Repo အား Clone ရန်

- မိမိ clone လုပ်မည့် directory တွင် CLI အားဖွင့်ပါ။
- CLI ဖွင့်ပြီး အောက်ပါအတိုင်း ရိုက်ထည့်ပါ။

```bash
git clone https://github.com/htetoowaiyan19/beacon-project
cd beacon-project
```

- ထို့နောက် Python Scripts များစမ်းလိုပါက requirements များ install ပါ

```bash
pip install -r requirements.txt
```

### ကိုယ်ပိုင် Model တစ်ခု train လိုပါက မိမိ PC နှင့်ကိုက်ညီသော Model ကိုရွေး train ပေးဖို့လိုပါတယ်။

| Model      | Parameters | Recommended GPU        | Notes                      |
| ---------- | ---------- | ---------------------- | -------------------------- |
| Qwen3-0.6B | 0.6B       | 6–8 GB VRAM            | Very fast, limited quality |
| Qwen3-1.7B | 1.7B       | 8–10 GB VRAM           | Good for experimentation   |
| Qwen3-4B   | 4B         | 12–16 GB VRAM          | Excellent balance          |
| Qwen3-8B   | 8B         | 16–24 GB VRAM          | Strong chatbot quality     |
| Qwen3-14B  | 14B        | 24–48 GB VRAM          | High-quality assistant     |
| Qwen3-32B  | 32B        | 48–80 GB VRAM          | Enterprise/research        |
| Qwen3-72B  | 72B        | Multi-GPU 160+ GB VRAM | Large-scale training       |
