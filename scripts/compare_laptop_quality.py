"""Compare existing GGUF smoke answers with the original trained adapter at the same output budget."""
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from backend.services.model_service import ModelService
    from sacrebleu.metrics import CHRF
    smoke = json.loads((ROOT / 'outputs/laptop_export/smoke.json').read_text(encoding='utf-8'))
    model = ModelService()
    scorer = CHRF()
    results = []
    for sample in smoke['results']:
        if sample['reference'] is None:
            continue
        generator = model.stream_chat(messages=[{'role': 'user', 'content': sample['prompt']}],
                                      temperature=0, max_new_tokens=192, think=False)
        chunks = []
        while True:
            try:
                chunks.append(next(generator))
            except StopIteration as finished:
                metrics = finished.value
                break
        answer = ''.join(chunks)
        base_score = scorer.sentence_score(answer, [sample['reference']]).score
        gguf_score = scorer.sentence_score(sample['answer'], [sample['reference']]).score
        results.append({**sample, 'original_trained_answer': answer,
                        'original_trained_metrics': metrics,
                        'original_trained_chrf': base_score, 'gguf_chrf': gguf_score})
        print(f"{sample['category']}: original chrF={base_score:.2f}, Q4 chrF={gguf_score:.2f}", flush=True)
    report = {'examples': len(results), 'original_trained_mean_chrf': statistics.mean(r['original_trained_chrf'] for r in results),
              'gguf_mean_chrf': statistics.mean(r['gguf_chrf'] for r in results),
              'scope': 'Eight seeded single-turn test examples, temperature 0, 192 output tokens. Not a full evaluation or native-language approval. Reference similarity does not establish accuracy.',
              'speed_scope': 'Different runtimes/devices (GGUF desktop CPU; original CUDA); not an M2 speed benchmark.',
              'results': results}
    (ROOT / 'outputs/laptop_export/quality-comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
