"""Merge the frozen adapter and export Q4_K_M GGUF using pinned llama.cpp tools."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.release import RELEASE, RELEASE_ADAPTER
from scripts.run_release import verify_release


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llama-source', type=Path, required=True)
    parser.add_argument('--quantizer', type=Path, required=True)
    parser.add_argument('--tool-tag', required=True)
    parser.add_argument('--work', type=Path, default=ROOT / 'outputs/laptop_export')
    parser.add_argument('--resume-conversion', action='store_true', help='Reuse a verified merged export after a converter failure')
    args = parser.parse_args()
    converter = args.llama_source.resolve() / 'convert_hf_to_gguf.py'
    quantizer = args.quantizer.resolve()
    if not converter.is_file() or not quantizer.is_file():
        parser.error('Provide the converter source directory and quantizer executable.')
    verify_release()
    output = ROOT / 'models/gguf'
    output.mkdir(parents=True, exist_ok=True)
    target = output / 'beacon-v1.0.0-Q4_K_M.gguf'
    if target.exists():
        parser.error('Export already exists; preserve or move it before creating a new export.')
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    merged = work / 'merged-hf'
    if merged.exists() and not args.resume_conversion:
        parser.error('Merged working directory already exists; use a new --work directory.')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    base_path = ROOT / 'models/qwen3-4b'
    receipt = merged / 'merge-receipt.json'
    if args.resume_conversion:
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['source_adapter_sha256'] != RELEASE['adapter_sha256']:
            raise ValueError('Merged export came from a different adapter')
        for name, expected in saved['files'].items():
            if not (merged / name).resolve().is_relative_to(merged) or digest(merged / name) != expected:
                raise ValueError(f'Merged export checksum mismatch: {name}')
    else:
        print('Merging the frozen trained adapter (original files remain unchanged)...', flush=True)
        tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_path, dtype=torch.bfloat16, device_map={'': 0} if torch.cuda.is_available() else None,
            local_files_only=True)
        model = PeftModel.from_pretrained(base, RELEASE_ADAPTER).merge_and_unload(safe_merge=True)
        model.save_pretrained(merged, safe_serialization=True, max_shard_size='5GB')
        tokenizer.save_pretrained(merged)
        receipt.write_text(json.dumps({'source_adapter_sha256': RELEASE['adapter_sha256'],
            'files': {p.name: digest(p) for p in merged.iterdir() if p.is_file()}}, indent=2), encoding='utf-8')
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    high_precision = work / 'beacon-v1.0.0-BF16.gguf'
    subprocess.run([sys.executable, str(converter), str(merged), '--outtype', 'bf16',
                    '--outfile', str(high_precision), '--model-name', 'BEACON v1.0.0 Burmese IT'], check=True)
    temporary = target.with_suffix('.gguf.tmp')
    try:
        subprocess.run([str(quantizer), str(high_precision), str(temporary), 'Q4_K_M'], check=True)
        with temporary.open('rb') as f:
            if f.read(4) != b'GGUF':
                raise ValueError('Export did not produce a GGUF model')
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {'release': RELEASE['version'], 'model_file': target.name,
                'quantization': 'Q4_K_M', 'adapter_merged': True,
                'source_adapter_sha256': RELEASE['adapter_sha256'],
                'source_base_sha256': digest(base_path / 'model.safetensors'),
                'sha256': digest(target), 'bytes': target.stat().st_size,
                'llama_cpp_tag': args.tool_tag,
                'converter_sha256': digest(converter), 'quantizer_sha256': digest(quantizer),
                'quality_status': 'Conversion complete; evaluation required before deployment.'}
    (output / 'export.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Exported {target} ({target.stat().st_size / 1024**3:.2f} GiB)', flush=True)


if __name__ == '__main__':
    main()
