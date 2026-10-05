"""Verify completed single-process Trainer checkpoints before recovery."""
import hashlib
import json
import os
from pathlib import Path

MARKER = 'checkpoint_complete.json'

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def seal_checkpoint(path):
    path = Path(path)
    required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin']
    if not all((path / name).is_file() for name in required):
        raise RuntimeError(f'Incomplete training state in {path}')
    if not any((path / name).is_file() for name in ('adapter_model.safetensors', 'model.safetensors')):
        raise RuntimeError(f'Model weights missing in {path}')
    files = {p.relative_to(path).as_posix(): digest(p) for p in path.rglob('*') if p.is_file() and p.name not in (MARKER, MARKER+'.tmp')}
    temporary = path / (MARKER + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(files, stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path / MARKER)

def valid_checkpoint(path):
    path = Path(path)
    try:
        files = json.loads((path / MARKER).read_text(encoding='utf-8'))
        required = {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'training_args.bin'}
        if not required.issubset(files) or not {'adapter_model.safetensors','model.safetensors'}.intersection(files): return False
        return all((path/name).resolve().is_relative_to(path.resolve()) and digest(path/name)==value for name,value in files.items())
    except (OSError, ValueError, TypeError):
        return False

def latest_checkpoint(output):
    candidates = [p for p in Path(output).glob('checkpoint-*') if p.is_dir() and p.name[11:].isdigit()]
    for path in sorted(candidates, key=lambda p:int(p.name[11:]), reverse=True):
        if valid_checkpoint(path): return path
    raise ValueError(f'No complete, verified checkpoint in {output}. Start a new run if none was saved.')
