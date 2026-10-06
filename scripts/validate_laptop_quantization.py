"""Small frozen-validation comparison of BF16, Q4 and Q5; does not train or edit data."""
import gc
import json
import os
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from sacrebleu.metrics import CHRF
    from backend.services.model_service import ModelService
    from backend.services.gguf_model_service import GGUFModelService
    rows = [json.loads(line) for line in (ROOT / 'datasets/releases/it_seminar_v3/validation/validation_combined.jsonl').read_text(encoding='utf-8').splitlines()]
    rng = random.Random(73)
    selected = []
    for tag in ('smalltalk_about_ai', 'NLP', 'computer hardware'):
        pool = sorted([r for r in rows if r['tags'] == tag], key=lambda r: r['id'])
        selected.append(rng.choice(pool))
    results = [{"id": row['id'], 'category': row['tags'],
                'prompt': next(m['content'] for m in row['messages'] if m['role'] == 'user'),
                'reference': next(m['content'] for m in row['messages'] if m['role'] == 'assistant'),
                'answers': {}, 'scores': {}} for row in selected]
    scorer = CHRF()
    for variant in ('BF16', 'Q4_K_M', 'Q5_K_M'):
        if variant != 'BF16':
            os.environ['BEACON_GGUF_METADATA'] = 'export.json' if variant == 'Q4_K_M' else 'export-Q5_K_M.json'
        model = ModelService() if variant == 'BF16' else GGUFModelService()
        try:
            for item in results:
                gen = model.stream_chat(messages=[{'role': 'user', 'content': item['prompt']}],
                                        temperature=0, max_new_tokens=192, think=False)
                answer = ''.join(gen)
                item['answers'][variant] = answer
                item['scores'][variant] = scorer.sentence_score(answer, [item['reference']]).score
                print(f"{variant} / {item['category']}: chrF {item['scores'][variant]:.2f}", flush=True)
        finally:
            if hasattr(model, 'close'):
                model.close()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    report = {'split': 'validation', 'seed': 73, 'examples': len(results),
              'mean_chrf': {v: statistics.mean(r['scores'][v] for r in results) for v in ('BF16', 'Q4_K_M', 'Q5_K_M')},
              'scope': 'Three diagnostic validation examples; not statistically reliable or native-speaker approval. No training or data changes. CPU GGUF and CUDA BF16 speed are not compared.',
              'results': results}
    (ROOT / 'outputs/laptop_export/validation-quantization.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
