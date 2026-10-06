"""Offline comparison of old history/streaming behavior and the bounded version."""
from datetime import datetime
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from transformers import TextIteratorStreamer
    from backend.services import model_service as module
    from backend.services.chat_service import ChatService
    from scripts.run_release import verify_release
    service = module.ModelService(lora_checkpoint_path=verify_release())
    service.load()
    original = (module.DEFAULT_MAX_PROMPT_TOKENS, module.DEFAULT_MAX_HISTORY_TURNS, module.UnicodeTextIteratorStreamer)
    history = []
    history_token_counts = []
    # Synthetic, no-space Burmese history; this benchmark does not use visitor chats or test answers.
    for number in range(7):
        answer = 'ကွန်ပျူတာကဒေတာကိုတွက်ချက်ပေးပါတယ်။ဒေတာတွေကိုမှတ်ဉာဏ်ထဲမှာသိမ်းထားနိုင်ပါတယ်။' * 2
        ids = service.tokenizer.encode(answer, add_special_tokens=False)
        assert len(ids) <= 512, 'Benchmark history must fit the ordinary reply limit'
        history_token_counts.append(len(ids))
        history.extend([{'role': 'user', 'content': 'ကွန်ပျူတာအကြောင်းရှင်းပြပေးပါ။'},
                        {'role': 'assistant', 'content': answer}])
    prompt = 'ဒီအကြောင်းကိုအတိုချုံးပြောပြပေးပါ။'
    chat = ChatService(service)
    # Warm CUDA before comparing; retain the same adapter, input, sampling and output limit.
    list(chat.stream_response('Say hello in English.', [], temperature=0, max_new_tokens=16))
    cases = []
    try:
        for name, budget, turns, streamer in (
            ('old_full_history_space_stream', 40000, 1000, TextIteratorStreamer),
            ('full_history_unicode_stream', 40000, 1000, original[2]),
            ('bounded_history_unicode_stream', original[0], original[1], original[2]),
        ):
            module.DEFAULT_MAX_PROMPT_TOKENS = budget
            module.DEFAULT_MAX_HISTORY_TURNS = turns
            module.UnicodeTextIteratorStreamer = streamer
            print('Starting', name, flush=True)
            events = []; started = time.perf_counter()
            for frame in chat.stream_response(prompt, history, temperature=0, max_new_tokens=64):
                if '[DONE]' not in frame: events.append(json.loads(frame[6:]))
            errors = [e for e in events if e['type'] == 'error']
            assert not errors, errors
            metrics = next(e['data'] for e in events if e['type'] == 'data-metrics')
            cases.append(dict(case=name, metrics=metrics,
                              elapsed_wall_seconds=round(time.perf_counter() - started, 3),
                              answer=''.join(e.get('delta', '') for e in events)))
            print(name, json.dumps(metrics), flush=True)
    finally:
        module.DEFAULT_MAX_PROMPT_TOKENS, module.DEFAULT_MAX_HISTORY_TURNS, module.UnicodeTextIteratorStreamer = original
    report = dict(description='Offline warmed-up 7-exchange synthetic-history comparison; single sample per case, not a latency guarantee.',
                  history_answer_tokens=history_token_counts, model=service.get_gpu_status(), cases=cases)
    directory = ROOT / 'outputs/performance'; directory.mkdir(parents=True, exist_ok=True)
    path = directory / datetime.now().strftime('chat_latency_%Y%m%d_%H%M%S.json')
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('REPORT', path, flush=True)


if __name__ == '__main__':
    main()
