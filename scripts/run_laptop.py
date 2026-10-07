"""Run the lightweight laptop edition with the existing members/visitor UI."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--check-runtime', action='store_true', help='Load the native model once, report status, then stop it')
    parser.add_argument('--chat', action='store_true', help='Run a single chat server instead of the control panel')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    os.environ['BEACON_RUNTIME'] = 'gguf'
    if args.cpu:
        os.environ['BEACON_GGUF_GPU_LAYERS'] = '0'
    from scripts.utils.llama_runtime import verify_model
    path, metadata = verify_model()
    print(f"Verified trained laptop model: {metadata['quantization']} / {path.name}", flush=True)
    if args.check:
        return
    if args.check_runtime:
        from backend.services.gguf_model_service import GGUFModelService
        import json
        service = GGUFModelService()
        try:
            service.load()
            print(json.dumps(service.get_gpu_status(), indent=2), flush=True)
        finally:
            service.close()
        return
    if args.chat:
        import uvicorn
        uvicorn.run('backend.app:app', host='127.0.0.1', port=8000)
    else:
        import tkinter as tk
        from scripts.show_day_ui import ShowDayUI
        window = tk.Tk()
        ShowDayUI(window)
        window.mainloop()


if __name__ == '__main__':
    main()
