"""Check the real trained GGUF through streaming; never changes model weights."""
import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.services.gguf_model_service import GGUFModelService
from backend.services.chat_service import ChatService


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--heldout', action='store_true', help='Also sample eight categories from the local test split')
    args = parser.parse_args()
    prompts = [('English', 'What is RAM? Answer briefly in English.', None),
               ('Burmese', 'TCP နဲ့ UDP ဘာကွာလဲ။ တိုတိုရှင်းပြပါ။', None)]
    if args.heldout:
        data = ROOT / 'datasets/releases/it_seminar_v3/test/test_combined.jsonl'
        rows = [json.loads(line) for line in data.read_text(encoding='utf-8').splitlines()]
        random.Random(42).shuffle(rows)
        categories = set()
        for row in rows:
            tag = row['tags']
            if tag in categories:
                continue
            categories.add(tag)
            question = next(m['content'] for m in row['messages'] if m['role'] == 'user')
            reference = next(m['content'] for m in row['messages'] if m['role'] == 'assistant')
            prompts.append((tag, question, reference))
            if len(categories) == 8:
                break
    service = GGUFModelService()
    results = []
    started = time.perf_counter()
    try:
        service.load()
        load_seconds = time.perf_counter() - started
        for category, prompt, reference in prompts:
            frames = []
            for raw in ChatService(service).stream_response(prompt, [], temperature=0, max_new_tokens=192):
                if '[DONE]' not in raw:
                    frames.append(json.loads(raw[6:]))
            errors = [p for p in frames if p['type'] == 'error']
            if errors:
                raise RuntimeError(errors)
            answer = ''.join(p.get('delta', '') for p in frames if p['type'] == 'text-delta')
            if not answer.strip() or '\ufffd' in answer:
                raise ValueError('Empty or damaged UTF-8 output')
            if category == 'English' and any('\u1000' <= c <= '\u109f' for c in answer):
                raise ValueError('English smoke reply did not follow English')
            if category == 'Burmese' and not any('\u1000' <= c <= '\u109f' for c in answer):
                raise ValueError('Burmese smoke reply did not contain Burmese')
            metrics = next(p['data'] for p in frames if p['type'] == 'data-metrics')
            result = {'category': category, 'prompt': prompt, 'answer': answer,
                      'reference': reference, 'metrics': metrics}
            results.append(result)
            print(f"{category}: first text {metrics['time_to_first_text_seconds']}s, "
                  f"{metrics['total_tokens']} tokens, {metrics['tokens_per_second']} tok/s", flush=True)
        report = {'runtime': service.get_gpu_status(), 'load_seconds': round(load_seconds, 3),
                  'hardware_scope': 'Executed on this machine; not an M2 laptop benchmark.',
                  'results': results, 'native_speaker_review': 'not performed'}
        directory = ROOT / 'outputs/laptop_export'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    finally:
        service.close()


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
