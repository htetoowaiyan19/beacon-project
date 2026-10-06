"""Dedicated, local backend controlled by the show-day desktop panel."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args()
    if not os.getenv('BEACON_MONITOR_TOKEN'):
        parser.exit(1, 'Start this backend from show_day_ui.bat.\n')
    os.environ['BEACON_SHOW_DAY'] = '1'
    if not args.preview:
        if os.getenv('BEACON_RUNTIME') != 'gguf':
            from scripts.run_release import verify_release
            try:
                os.environ['BEACON_LORA_PATH'] = str(verify_release())
            except ValueError as exc:
                parser.exit(1, f'{exc}\n')
        from backend.app import app
        print('Loading the trained seminar model before accepting visitors…', flush=True)
        app.state.chat_service.model_service.load()
        print('Trained model ready for the visitor frontend.', flush=True)
    else:
        from backend.dev_app import app
        print('FRONTEND PREVIEW: synthetic replies; no model is loaded.', flush=True)
    import uvicorn
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port,
                                          timeout_graceful_shutdown=5))
    app.state.show_day_shutdown = lambda: setattr(server, 'should_exit', True)
    server.run()


if __name__ == '__main__':
    main()
