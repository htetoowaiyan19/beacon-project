"""Use a tiny PEFT model to verify base isolation, actual token counts and worker errors."""
import pytest


def make_service():
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    from backend.services.model_service import ModelService
    service = ModelService()
    service.tokenizer = AutoTokenizer.from_pretrained(str(service.base_model_path), local_files_only=True)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(service.tokenizer), hidden_size=32,
        intermediate_size=64, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16))
    service.base_model = model
    service.lora_model = get_peft_model(model, LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj"], task_type="CAUSAL_LM"))
    service._is_loaded = True
    return service


def exhaust(generator):
    chunks = []
    while True:
        try:
            chunks.append(next(generator))
        except StopIteration as done:
            return chunks, done.value


def test_base_stream_disables_adapter_and_counts_generated_ids(monkeypatch):
    import torch
    service = make_service()
    observed = []
    def generate(input_ids, streamer, **kwargs):
        assert kwargs['use_cache'] is True
        observed.append(service.base_model.model.layers[0].self_attn.q_proj.disable_adapters)
        # A single decoded text chunk can contain multiple generated tokens.
        tokens = service.tokenizer.encode("hello world", add_special_tokens=False)
        streamer.put(input_ids)
        streamer.put(torch.tensor(tokens))
        streamer.end()
        return torch.cat([input_ids, torch.tensor([tokens])], dim=1)
    monkeypatch.setattr(service.base_model, "generate", generate)
    _, metrics = exhaust(service.stream_chat(messages=[{"role": "user", "content": "hi"}], use_base_model=True))
    assert observed == [True]
    assert metrics["total_tokens"] == len(service.tokenizer.encode("hello world", add_special_tokens=False))
    assert service.base_model.model.layers[0].self_attn.q_proj.disable_adapters is False


def test_worker_failure_propagates(monkeypatch):
    service = make_service()
    def fail(**kwargs):
        raise ValueError("broken generation")
    monkeypatch.setattr(service.base_model, "generate", fail)
    with pytest.raises(RuntimeError, match="generation failed"):
        exhaust(service.stream_chat(messages=[{"role": "user", "content": "hi"}], use_base_model=True))
