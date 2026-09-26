"""One policy per server, with an exclusive episode lease to prevent state mixing."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import logging
import threading
import time

from redvla.protocol import CONVENTION, MAX_BYTES, VERSION, decode_observation, validate_actions

LOG = logging.getLogger(__name__)


def make_server(model, host="127.0.0.1", port=8001, token=None, lease_seconds=3600):
    lock = threading.Lock()
    owner = None
    last_seen = 0.0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)

        def setup(self):
            super().setup()
            self.connection.settimeout(120)

        def reply(self, code, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            if token and not hmac.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                self.reply(401, {"error": "Invalid bearer token"})
                return False
            return True

        def do_GET(self):
            if not self.authorized():
                return
            if self.path != "/health":
                self.reply(404, {"error": "Unknown endpoint"})
                return
            self.reply(200, {"protocol": VERSION, "convention": CONVENTION,
                             "model": type(model).__name__, "ready": True,
                             "policy": getattr(model, "metadata", {})})

        def do_POST(self):
            nonlocal owner, last_seen
            if not self.authorized():
                return
            if self.path not in ("/reset", "/predict", "/close"):
                self.reply(404, {"error": "Unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= MAX_BYTES:
                    raise ValueError("Invalid request size")
                body = self.rfile.read(size)
                if len(body) != size:
                    raise ValueError("Incomplete request")
                request = json.loads(body)
                if not isinstance(request, dict) or request.get("protocol") != VERSION:
                    raise ValueError("Unsupported protocol version")
                session = request.get("session")
                if not isinstance(session, str) or not 1 <= len(session) <= 128:
                    raise ValueError("Missing or invalid session")
                if self.path == "/predict":
                    obs = decode_observation(request["observation"])
                    instruction = request["instruction"]
                    context = request.get("context", {})
                    if not isinstance(instruction, str) or not isinstance(context, dict):
                        raise ValueError("Invalid instruction or context")
            except (ValueError, KeyError, TypeError, OverflowError) as error:
                self.reply(400, {"error": str(error)})
                return
            with lock:
                if owner is not None and time.monotonic() - last_seen > lease_seconds:
                    owner = None
                if owner not in (None, session):
                    self.reply(409, {"error": "Policy is leased by another evaluator; use one server per worker"})
                    return
                try:
                    if self.path == "/reset":
                        model.prepare_for_new_episode()
                        owner = session
                        result = {"ok": True}
                    elif self.path == "/close":
                        owner = None
                        result = {"ok": True}
                    else:
                        if owner != session:
                            self.reply(409, {"error": "Reset required before prediction"})
                            return
                        model.set_inference_context(**context)
                        result = {"actions": validate_actions(model.predict(obs, instruction)).tolist()}
                    last_seen = time.monotonic()
                    self.reply(200, result)
                except Exception as error:
                    LOG.exception("Policy request failed")
                    self.reply(500, {"error": f"{type(error).__name__}: {error}"})

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = False
    return server
