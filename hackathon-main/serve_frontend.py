from __future__ import annotations

import os
import socket
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "frontend"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_PORT_TRIES = 20


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)


def is_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_free_port(host: str = DEFAULT_HOST) -> int:
    requested = int(os.environ.get("FRONTEND_PORT", DEFAULT_PORT))
    for port in range(requested, requested + MAX_PORT_TRIES):
        if is_port_available(host, port):
            return port
    raise RuntimeError(
        f"No free localhost port found in range {requested}-{requested + MAX_PORT_TRIES - 1}."
    )


if __name__ == "__main__":
    host = os.environ.get("FRONTEND_HOST", DEFAULT_HOST)
    port = find_free_port(host)
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Frontend: http://{host}:{port}")
    print("Press Ctrl+C to stop the frontend server.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFrontend stopped.")
    finally:
        server.server_close()
