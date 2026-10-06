"""Regression checks for growing context and space-dependent Burmese streaming."""
import json
from pathlib import Path
import pytest

from backend.services.chat_context import prepare_chat, ChatContextError
from backend.services.chat_service import ChatService


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    root = Path(__file__).resolve().parents[1]
    return AutoTokenizer.from_pretrained(str(root / 'models/qwen3-4b'), local_files_only=True)


def test_token_budget_preserves_current_input_and_recent_complete_exchanges(tokenizer):
    messages = []
    for number in range(7):
        messages.extend([{'role': 'user', 'content': f'Question {number}'},
                         {'role': 'assistant', 'content': 'ကွန်ပျူတာကဒေတာကိုတွက်ချက်ပေးပါတယ်။' * 18}])
    current = {'role': 'user', 'content': 'နောက်တစ်ခုကိုရှင်းပြပေးပါ။'}
    messages.append(current)
    original = json.dumps(messages, ensure_ascii=False)
    encoded, conversation, metrics = prepare_chat(tokenizer, messages, 'Be helpful.', max_prompt_tokens=2048)
    assert encoded['input_ids'].shape[-1] <= 2048
    assert conversation[0] == {'role': 'system', 'content': 'Be helpful.'}
    assert conversation[-1] == current
    history = conversation[1:-1]
    assert len(history) % 2 == 0
    assert [m['role'] for m in history] == ['user', 'assistant'] * (len(history) // 2)
    assert metrics['history_turns_used'] <= 3 and metrics['history_messages_dropped'] >= 8
    if history: assert history[-2]['content'] == 'Question 6'
    assert json.dumps(messages, ensure_ascii=False) == original


def test_native_context_reserves_output_budget(tokenizer):
    messages = [{'role': 'user', 'content': 'Older question'}, {'role': 'assistant', 'content': 'large answer ' * 100},
                {'role': 'user', 'content': 'Hi'}]
    encoded, conversation, metrics = prepare_chat(tokenizer, messages, 'Be helpful.', context_limit=128, max_new_tokens=64)
    assert metrics['prompt_token_budget'] == 64
    assert encoded['input_ids'].shape[-1] + 64 <= 128
    assert conversation[-1]['content'] == 'Hi'


def test_large_current_message_is_rejected_instead_of_silently_cut(tokenizer):
    with pytest.raises(ChatContextError, match='Shorten'):
        prepare_chat(tokenizer, [{'role': 'user', 'content': 'မြန်မာစာ' * 200}], 'Be helpful.', max_prompt_tokens=128)


@pytest.mark.parametrize('text', ['မင်္ဂလာပါ။ဒီနေ့နေကောင်းလား။', 'hello👩🏽‍💻မြန်မာစာ🙂world'])
def test_unicode_streams_before_spaces_and_preserves_utf8(tokenizer, text):
    import torch
    from transformers import TextIteratorStreamer
    from backend.services.text_streamer import UnicodeTextIteratorStreamer
    stream = UnicodeTextIteratorStreamer(tokenizer, skip_special_tokens=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    first_at = None
    for number, token in enumerate(tokens):
        stream.put(torch.tensor([token]))
        while not stream.text_queue.empty():
            chunk = stream.text_queue.get_nowait()
            assert '\ufffd' not in chunk
            chunks.append(chunk)
            if first_at is None and chunk: first_at = number
    assert first_at is not None and first_at < len(tokens) - 1
    stream.end()
    while not stream.text_queue.empty():
        chunk = stream.text_queue.get_nowait()
        if chunk is not None: chunks.append(chunk)
    assert ''.join(chunks) == tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    if ' ' not in text and not text.startswith('hello'):
        original = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
        for token in tokens: original.put(torch.tensor([token]))
        assert not any(original.text_queue.get_nowait() for _ in range(original.text_queue.qsize()))


def test_context_error_is_actionable_and_metrics_measure_first_text():
    class Model:
        def stream_chat(self, **kwargs):
            yield 'မင်္ဂလာပါ'
            return {'total_tokens': 5, 'finish_reason': 'stop', 'queue_seconds': 0}
    parts = [json.loads(f[6:]) for f in ChatService(Model()).stream_response('Hi', []) if '[DONE]' not in f]
    metrics = next(e['data'] for e in parts if e['type'] == 'data-metrics')
    assert metrics['time_to_first_text_seconds'] >= 0
    assert metrics['request_elapsed_seconds'] >= metrics['time_to_first_text_seconds']
    class TooLong:
        def stream_chat(self, **kwargs):
            raise ChatContextError('Shorten your message.')
            yield ''
    parts = [json.loads(f[6:]) for f in ChatService(TooLong()).stream_response('Hi', []) if '[DONE]' not in f]
    assert next(e for e in parts if e['type'] == 'error')['errorText'] == 'Shorten your message.'
