import socket
import threading
import time
import msgpack
import numpy as np
import pytest
from websockets.sync.server import serve
from redvla.core.types import Observation
from redvla.models.openpi import OpenPIModel, _pack, _unpack


def test_native_openpi_handshake_and_action_chunk(monkeypatch):
    received = []
    def handler(ws):
        ws.send(msgpack.packb({"model": "protocol-fixture"}))
        message = msgpack.unpackb(ws.recv(), object_hook=_unpack)
        received.append(message)
        ws.send(msgpack.packb({"actions": np.ones((16, 7), np.float32)}, default=_pack))
    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.socket.getsockname()[1]
        client = OpenPIModel(replan_steps=4)
        client.load(f"ws://127.0.0.1:{port}")
        obs = Observation(np.full((256, 256, 3), 27, np.uint8), np.full((256, 256, 3), 45, np.uint8), np.ones(8))
        try:
            actions = client.predict(obs, "put the bowl on the plate")
            assert np.shape(actions) == (4, 7)
            assert received[0]["prompt"] == "put the bowl on the plate"
            assert received[0]["observation/image"].shape == (224, 224, 3)
            assert received[0]["observation/state"].dtype == np.float32
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=2)


def test_openpi_connect_timeout_is_bounded():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        OpenPIModel(connect_timeout=0.15).load(f"ws://127.0.0.1:{port}")
    assert time.monotonic() - start < 2
