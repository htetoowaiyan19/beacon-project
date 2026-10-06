"""Independent visitor server with a restricted streaming proxy to the backend."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
GET_API = {'/api/health', '/api/config'}
POST_API = {'/api/chat/stream', '/api/visitor/event'}
ASSETS = {'/app.js', '/stream.js', '/show-day.js', '/style.css', '/favicon.svg'}


class VisitorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, backend_url, **kwargs):
        self.backend_url = backend_url
        super().__init__(*args, directory=str(ROOT / 'frontend'), **kwargs)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in GET_API:
            return self.proxy('GET')
        if path in ('/', '/index.html'):
            html = (ROOT / 'frontend/index.html').read_text(encoding='utf-8')
            payload = html.replace('<body>', '<body class="visitor-mode">', 1).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers(); self.wfile.write(payload)
        elif path in ASSETS:
            super().do_GET()
        else:
            self.send_error(404, 'Visitor route unavailable')

    def do_HEAD(self):
        if urlsplit(self.path).path not in ASSETS:
            return self.send_error(404)
        super().do_HEAD()

    def do_POST(self):
        if urlsplit(self.path).path not in POST_API:
            return self.send_error(404, 'Visitor route unavailable')
        self.proxy('POST')

    def proxy(self, method):
        try:
            size = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            return self.send_error(400, 'Invalid content length')
        if size < 0 or size > 1024 * 1024:
            return self.send_error(413, 'Conversation is too large; start a new chat')
        self.connection.settimeout(30)
        body = self.rfile.read(size) if method == 'POST' else None
        # The visitor server never forwards operator routes or authentication headers.
        request = Request(self.backend_url + urlsplit(self.path).path, data=body, method=method,
                          headers={'Content-Type': 'application/json'})
        try:
            upstream = urlopen(request, timeout=180)
        except HTTPError as exc:
            upstream = exc
        except (URLError, TimeoutError, OSError):
            return self.send_error(503, 'Backend is offline. Ask a seminar member to start it.')
        try:
            with upstream:
                self.send_response(upstream.status)
                for name in ('Content-Type', 'x-vercel-ai-ui-message-stream', 'Cache-Control'):
                    if upstream.headers.get(name):
                        self.send_header(name, upstream.headers[name])
                self.send_header('Connection', 'close')
                self.send_header('X-Accel-Buffering', 'no')
                self.end_headers()
                self.close_connection = True
                while chunk := upstream.read1(8192):
                    self.wfile.write(chunk); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            # Closing upstream lets the backend cancel an abandoned response.
            self.close_connection = True


def make_server(port=8080, backend_port=8000):
    return ThreadingHTTPServer(('127.0.0.1', port),
                              partial(VisitorHandler, backend_url=f'http://127.0.0.1:{backend_port}'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--backend-port', type=int, default=8000)
    args = parser.parse_args()
    server = make_server(args.port, args.backend_port)
    print(f'Visitor page: http://127.0.0.1:{args.port}/', flush=True)
    print(f'Model backend: http://127.0.0.1:{args.backend_port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
