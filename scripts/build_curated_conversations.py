"""Compatibility entry point for the individually authored fluency seed."""
from pathlib import Path
if __package__:
    from .build_fluency_dataset import build, ROOT
else:
    from build_fluency_dataset import build, ROOT

if __name__ == '__main__':
    print(build(tokenizer_path=ROOT / 'models/qwen3-4b'))
