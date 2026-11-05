#!/usr/bin/env python3
"""
serve.py  --  ONE persistent local process that holds the wind, T and RH models (pipeline/worker/handler.py) and answers
POST /run {"input": {...}} one request at a time (a lock), GET /health.   Started on demand by ai/backend.py, stops after
--idle seconds without a request (default 3600) or with  kill $(cat <state>.pid) .

    python ai/serve.py --pipe pipeline --state .ai_server/state.json [--port 0] [--idle 3600]
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipe", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--idle", type=float, default=3600.0)
    ap.add_argument("--mock", action="store_true", help="serve ai/mock_models.py instead of the real models (plumbing test)")
    a = ap.parse_args()
    pipe = Path(a.pipe).resolve()
    os.environ["APP_ROOT"] = str(pipe)
    sys.path.insert(0, str(pipe))
    if a.mock:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import mock_models as H
        handler = H.handler
    else:
        import importlib
        H = importlib.import_module("worker.handler")      # namespace package: pipeline/worker/handler.py
        H._warmup()
        handler = H.handler
    lock = threading.Lock()
    last = [time.time()]

    class Req(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": True})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                event = json.loads(self.rfile.read(n))
            except ValueError:
                return self._send(400, {"error": "bad json"})
            with lock:
                last[0] = time.time()
                try:
                    out = handler(event)
                except Exception as exc:  # noqa: BLE001
                    out = {"error": f"{type(exc).__name__}: {exc}"}
                last[0] = time.time()
            self._send(200, out)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Req)
    state = Path(a.state)
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps({"port": srv.server_address[1], "pid": os.getpid(), "started": time.time(), "mock": a.mock}))
    Path(str(state) + ".pid").write_text(str(os.getpid()))
    print(f"[serve] ready on 127.0.0.1:{srv.server_address[1]} (pid {os.getpid()})", flush=True)

    def watchdog():
        while True:
            time.sleep(30)
            if not lock.locked() and time.time() - last[0] > a.idle:
                print("[serve] idle, stopping", flush=True)
                state.unlink(missing_ok=True)
                os._exit(0)
    threading.Thread(target=watchdog, daemon=True).start()
    srv.serve_forever()


if __name__ == "__main__":
    main()
