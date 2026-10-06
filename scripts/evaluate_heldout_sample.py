"""Reproducible stratified held-out generation comparison; never updates weights."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from evaluate_model import generate_response, score_reference, get_inference_dtype
from utils.active_dataset import active_release, ROOT
from utils.dataset_checks import assert_disjoint


def digest(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def bucket(row):
    tag=row['tags']
    if tag=='computer hardware': return 'Hardware'
    if tag=='Operating System': return 'Operating systems'
    if tag=='NLP': return 'NLP'
    if tag=='networking_cryptography': return 'Cryptography'
    if tag=='cybersecurity basics': return 'Cybersecurity'
    if tag.startswith(('network','transport')) or tag=='Computer Networking': return 'Networking'
    if tag.startswith(('smalltalk','authored')): return 'Companion styles'
    return 'Other conversation'


def main():
    # Fail before GPU loading if the declared scoring dependency is unavailable.
    from sacrebleu.metrics import CHRF
    parser=argparse.ArgumentParser()
    parser.add_argument('--adapter',required=True)
    args=parser.parse_args()
    adapter=Path(args.adapter).resolve()
    weights=adapter/'adapter_model.safetensors'
    if not weights.is_file(): raise FileNotFoundError(weights)
    release=active_release()
    paths={s:release/s/f'{s}_combined.jsonl' for s in ('train','validation','test')}
    splits={s:[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()] for s,p in paths.items()}
    assert_disjoint(splits['train'],splits['validation'],splits['test'])
    groups=defaultdict(list)
    for row in splits['test']: groups[bucket(row)].append(row)
    rng=random.Random(42);selected=[]
    for category in sorted(groups):
        pool=sorted(groups[category],key=lambda r:r['id']);rng.shuffle(pool)
        chosen=pool[:4]
        if category=='Companion styles':
            multi=next((r for r in pool if len(r['messages'])>3),None)
            if multi and multi not in chosen:chosen[-1]=multi
        selected.extend(chosen)
    directory=ROOT/'outputs/evaluations'/datetime.now(ZoneInfo('Asia/Rangoon')).strftime('heldout_%Y%m%d_%H%M%S')
    directory.mkdir(parents=True)
    manifest=dict(adapter=str(adapter),adapter_sha256=digest(weights),
        split_sha256={s:digest(p) for s,p in paths.items()},seed=42,thinking=False,
        sample_method='Four seeded random examples per broad category; ensure one multi-turn companion example. Macro-balanced, not population-weighted.',
        ids=[r['id'] for r in selected],categories=dict(Counter(bucket(r) for r in selected)),
        max_new_tokens=512,do_sample=True,temperature=.7,top_p=.8,repetition_penalty=1.15,
        full_test_count=len(splits['test']))
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (directory/'sample.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected),encoding='utf-8')
    print('RESULT_DIRECTORY',directory,flush=True)
    model_path=ROOT/'models/qwen3-4b'
    tokenizer=AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    base=AutoModelForCausalLM.from_pretrained(model_path,dtype=get_inference_dtype(),attn_implementation='sdpa',device_map={'':0},local_files_only=True)
    model=PeftModel.from_pretrained(base,adapter,local_files_only=True);model.eval()
    results=[]
    for n,row in enumerate(selected,1):
        messages=row['messages'][:-1];reference=row['messages'][-1]['content']
        item=dict(id=row['id'],category=bucket(row),messages=messages,reference=reference)
        for label in ('base','trained'):
            from contextlib import nullcontext
            with model.disable_adapter() if label=='base' else nullcontext():
                answer,tokens,seconds,tps=generate_response(model,tokenizer,messages[-1]['content'],False,messages)
            item[label]=dict(answer=answer,tokens=tokens,seconds=seconds,tokens_per_second=tps,
                             reached_token_budget=tokens>=512,empty=not answer.strip(),**score_reference(answer,reference))
        results.append(item)
        with (directory/'results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        print(f'{n}/{len(selected)} {row["id"]} base={item["base"].get("chrf")} trained={item["trained"].get("chrf")}',flush=True)
    summary={'count':len(results),'note':'Reference agreement is not factual accuracy; human review is required.', 'models':{}}
    for label in ('base','trained'):
        metrics=[r[label] for r in results]
        summary['models'][label]={k:sum(m[k] for m in metrics)/len(metrics) for k in ('normalized_exact_match','chrf') if all(k in m for m in metrics)}
        summary['models'][label].update(empty_answers=sum(m['empty'] for m in metrics),token_budget_hits=sum(m['reached_token_budget'] for m in metrics),tokens_per_second=sum(m['tokens'] for m in metrics)/sum(m['seconds'] for m in metrics))
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    lines=['# Held-out comparison','',json.dumps(summary,indent=2),'']
    for r in results:
        lines += [f'## {r["id"]} — {r["category"]}','',f'**Prompt:** {r["messages"][-1]["content"]}','',f'**Reference:** {r["reference"]}','',f'**Base:** {r["base"]["answer"]}','',f'**Trained:** {r["trained"]["answer"]}','']
    (directory/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
