#!/usr/bin/env python3
"""Start the Aesteel diagnostic app with one command.

Starts the FastAPI backend and the static frontend, finds a free frontend
port, opens the browser, and shuts both child processes down on Ctrl+C.
The backend intentionally stays on port 8000 because the frontend defaults to it.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HOST = os.environ.get("FRONTEND_HOST", "127.0.0.1")
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8000
FRONTEND_START = int(os.environ.get("FRONTEND_PORT", "8080"))
PORT_TRIES = 20


def port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def find_frontend_port() -> int:
    for port in range(FRONTEND_START, FRONTEND_START + PORT_TRIES):
        if port_available(HOST, port):
            return port
    raise RuntimeError(
        f"No free frontend port in {FRONTEND_START}-{FRONTEND_START + PORT_TRIES - 1}."
    )


def backend_is_aesteel() -> bool:
    try:
        url = f"http://{BACKEND_HOST}:{BACKEND_PORT}/api/health"
        with urllib.request.urlopen(url, timeout=1.5) as response:
            return response.status == 200 and "api" in response.read().decode("utf-8", "ignore")
    except (OSError, urllib.error.URLError, TimeoutError):
        return False


def start_backend() -> subprocess.Popen[bytes] | None:
    if not port_available(BACKEND_HOST, BACKEND_PORT):
        if backend_is_aesteel():
            print(f"Backend already running: http://{BACKEND_HOST}:{BACKEND_PORT}")
            return None
        raise RuntimeError(
            "Port 8000 is busy and does not look like the Aesteel API. "
            "Stop the process using port 8000 and run start.py again."
        )
    return subprocess.Popen([sys.executable, "-m", "backend"], cwd=ROOT)


def main() -> int:
    backend = None
    frontend = None
    try:
        if not (ROOT / "data" / "train.csv").exists() or not (ROOT / "data" / "val.csv").exists():
            raise RuntimeError("Missing data/train.csv or data/val.csv. Check the repository checkout.")

        backend = start_backend()
        if backend is not None:
            print(f"Backend starting: http://{BACKEND_HOST}:{BACKEND_PORT}")
            time.sleep(0.8)

        frontend_port = find_frontend_port()
        env = os.environ.copy()
        env["FRONTEND_HOST"] = HOST
        env["FRONTEND_PORT"] = str(frontend_port)
        frontend = subprocess.Popen([sys.executable, "serve_frontend.py"], cwd=ROOT, env=env)

        frontend_url = f"http://{HOST}:{frontend_port}"
        print(f"Frontend: {frontend_url}")
        print("Opening browser…")
        time.sleep(0.5)
        webbrowser.open(frontend_url)
        print("Press Ctrl+C to stop the application.")

        while True:
            if backend is not None and backend.poll() is not None:
                raise RuntimeError(f"Backend stopped unexpectedly (exit code {backend.returncode}).")
            if frontend.poll() is not None:
                raise RuntimeError(f"Frontend stopped unexpectedly (exit code {frontend.returncode}).")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopping Aesteel diagnostics…")
        return 0
    except Exception as exc:
        print(f"Startup error: {exc}", file=sys.stderr)
        return 1
    finally:
        for process in (frontend, backend):
            if process is not None and process.poll() is None:
                process.terminate()
        for process in (frontend, backend):
            if process is not None:
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
