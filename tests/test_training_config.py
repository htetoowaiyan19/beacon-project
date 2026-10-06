"""Check the real installed TRL trainer masks and a tiny model's loss without loading 4B weights."""
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_assistant_mask_and_chunked_loss(tmp_path):
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from trl import SFTTrainer
    from train_lora import build_training_arguments
    tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1] / "models/qwen3-4b"), local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = Qwen3ForCausalLM(Qwen3Config(
        vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    ))
    rows = Dataset.from_list([{"messages": [
        {"role": "user", "content": "Say hello"}, {"role": "assistant", "content": "မင်္ဂလာပါ"}
    ]}])
    config = build_training_arguments(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        assistant_only_loss=True, max_length=128, packing=False, loss_type="chunked_nll", report_to="none",
    )
    trainer = SFTTrainer(model=model, train_dataset=rows, args=config, processing_class=tokenizer)
    batch = trainer.data_collator([trainer.train_dataset[0]])
    labels = batch["labels"]
    assert (labels == -100).any() and (labels != -100).any()
    supervised = tokenizer.decode(labels[labels != -100].tolist())
    assert "မင်္ဂလာပါ" in supervised and "Say hello" not in supervised
    assert tokenizer.eos_token in supervised
    loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())


def test_unsupported_training_options_fail():
    from train_lora import build_training_arguments
    with pytest.raises(RuntimeError, match="required training options"):
        build_training_arguments(nonexistent_training_option=True)


def test_production_training_arguments_match_installed_trl():
    import ast
    import inspect
    from trl import SFTConfig
    path = Path(__file__).resolve().parents[1] / "scripts/train_lora.py"
    module = ast.parse(path.read_text(encoding="utf-8"))
    call = next(n for n in ast.walk(module) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "build_training_arguments")
    # **training_schedule(...) is an expansion, not an option named None.
    # Its evaluated schedule is checked by test_production_training_schedule_validates.
    assert not ({kw.arg for kw in call.keywords if kw.arg is not None}
                - set(inspect.signature(SFTConfig).parameters))
