"""Rehearse real CUDA GGUF conversations, record timings, and stop the model."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def driver_status():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                timeout=5, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        fields = result.stdout.splitlines()[0].split(',')
        return {'name': fields[0].strip(), 'total_mib': float(fields[1]), 'used_mib': float(fields[2]),
                'utilization_percent': float(fields[3])}
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--turns', type=int, default=8)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if not 2 <= args.turns <= 20:
        parser.error('Choose between 2 and 20 turns.')
    os.environ['BEACON_RUNTIME'] = 'gguf'
    if args.cpu:
        os.environ['BEACON_GGUF_GPU_LAYERS'] = '0'
    from backend.services.gguf_model_service import GGUFModelService
    from backend.services.chat_service import ChatService
    service = GGUFModelService()
    history = []
    questions = [('English', 'What is RAM? Answer briefly in English.'),
                 ('Burmese', 'TCP နဲ့ UDP ဘာကွာလဲ။ တိုတိုရှင်းပြပါ။'),
                 ('English', 'What is SSH used for? Answer briefly in English.'),
                 ('Burmese', 'CPU နဲ့ RAM ဘယ်လိုကွာခြားလဲ။ တိုတိုရှင်းပြပါ။')]
    results = []
    baseline = driver_status()
    started = time.perf_counter()
    try:
        service.load()
        load_seconds = round(time.perf_counter() - started, 3)
        if not args.cpu and not service.runtime.cuda:
            raise RuntimeError('CUDA was not selected. Use the Windows package/runtime.')
        for turn in range(args.turns):
            language, question = questions[turn % len(questions)]
            frames = []
            for raw in ChatService(service).stream_response(question, history, temperature=0, max_new_tokens=192):
                if '[DONE]' not in raw:
                    frames.append(json.loads(raw[6:]))
            errors = [frame for frame in frames if frame['type'] == 'error']
            if errors:
                raise RuntimeError(errors)
            answer = ''.join(frame.get('delta', '') for frame in frames if frame['type'] == 'text-delta')
            burmese = any('\u1000' <= char <= '\u109f' for char in answer)
            if not answer.strip() or '\ufffd' in answer or (language == 'English' and burmese) or (language == 'Burmese' and not burmese):
                raise ValueError('Empty/damaged output or wrong language at turn ' + str(turn + 1))
            metrics = next(frame['data'] for frame in frames if frame['type'] == 'data-metrics')
            if metrics['effective_max_new_tokens'] != 192 or metrics['history_turns_used'] > 2:
                raise ValueError('Laptop generation limits changed')
            history.extend([{'role': 'user', 'content': question}, {'role': 'assistant', 'content': answer}])
            results.append({'turn': turn + 1, 'language': language, 'question': question, 'answer': answer,
                            'metrics': metrics, 'driver': driver_status()})
            print(f"Turn {turn + 1}: {language}, first text {metrics['time_to_first_text_seconds']}s, "
                  f"{metrics['tokens_per_second']} tokens/s", flush=True)
        status = service.get_gpu_status()
    finally:
        service.close()
    report = {'hardware_scope': 'Measured on this device, not necessarily the RTX 4050 laptop.',
              'load_seconds': load_seconds, 'baseline_driver': baseline, 'runtime': status,
              'results': results, 'native_process_stopped': service.runtime.process.poll() is not None,
              'quality_scope': 'Stream/language/limits checks only; native Burmese and factual review required.'}
    directory = ROOT / 'outputs/windows_gpu_checks'
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / ('smoke-cpu.json' if args.cpu else 'smoke.json')
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Report: ' + str(destination), flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
