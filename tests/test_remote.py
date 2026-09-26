import threading
import numpy as np
import pytest
from redvla.core.types import Observation
from redvla.models.base import PolicyModel
from redvla.models.remote import RemoteModel
from redvla.serving import make_server


class StatefulPolicy(PolicyModel):
    def __init__(self):
        self.resets = 0

    def load(self, model_path):
        pass

    def prepare_for_new_episode(self):
        self.resets += 1

    def set_inference_context(self, **context):
        self.context = context

    def predict(self, obs, task_description):
        self.obs, self.prompt = obs, task_description
        return [np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32) for _ in range(2)]


@pytest.fixture
def endpoint():
    policy = StatefulPolicy()
    server = make_server(policy, port=0, token="test-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", policy
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_http_roundtrip_reset_and_exclusive_session(endpoint, monkeypatch):
    url, policy = endpoint
    monkeypatch.setenv("REDVLA_API_TOKEN", "test-token")
    client = RemoteModel()
    second = RemoteModel()
    client.load(url)
    second.load(url)
    obs = Observation(np.full((6, 7, 3), 42, np.uint8), np.zeros((4, 5, 3), np.uint8), np.arange(8, dtype=np.float32))
    try:
        client.prepare_for_new_episode()
        client.set_inference_context(marker="one-shot")
        actions = client.predict(obs, "move the bowl")
        assert len(actions) == 2
        np.testing.assert_array_equal(policy.obs.image, obs.image)
        np.testing.assert_array_equal(policy.obs.state, obs.state)
        assert policy.prompt == "move the bowl"
        assert policy.context == {"marker": "one-shot"}
        client.predict(obs, "move the bowl")
        assert policy.context == {}
        with pytest.raises(RuntimeError, match="409"):
            second.prepare_for_new_episode()
        client.prepare_for_new_episode()
        assert policy.resets == 2
        client.close()
        second.prepare_for_new_episode()
        assert policy.resets == 3
    finally:
        client.close()
        second.close()


def test_authorization_and_protocol_rejection(endpoint, monkeypatch):
    url, _ = endpoint
    monkeypatch.delenv("REDVLA_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="401"):
        RemoteModel().load(url)
    monkeypatch.setenv("REDVLA_API_TOKEN", "test-token")
    client = RemoteModel()
    client.load(url)
    with pytest.raises(RuntimeError, match="400"):
        client._request("/reset", {"protocol": 999, "session": "session"})
