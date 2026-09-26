"""HTTP client; the model lives in another process / Python environment."""
import json
import os
import uuid
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from redvla.models.base import PolicyModel
from redvla.protocol import CONVENTION, MAX_BYTES, VERSION, encode_observation, validate_actions


class RemoteModel(PolicyModel):
    def __init__(self, timeout=120.0, token_env="REDVLA_API_TOKEN"):
        self.timeout = timeout
        self.token_env = token_env
        self.url = None
        self.session = uuid.uuid4().hex
        self.active = False
        self.context = {}

    def _request(self, route, data=None):
        headers = {"Content-Type": "application/json"}
        token = os.environ.get(self.token_env)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = None if data is None else json.dumps(data, allow_nan=False).encode()
        request = Request(self.url + route, data=payload, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read(MAX_BYTES + 1)
                if len(body) > MAX_BYTES:
                    raise RuntimeError("Model server response exceeds transport limit")
                return json.loads(body)
        except HTTPError as error:
            detail = error.read(4096).decode(errors="replace")
            raise RuntimeError(f"Model server HTTP {error.code}: {detail}") from error

    def load(self, model_path):
        if not model_path.startswith(("http://", "https://")):
            raise ValueError("remote model path must be an http(s) URL")
        self.url = model_path.rstrip("/")
        metadata = self._request("/health")
        self.metadata = metadata
        if metadata.get("protocol") != VERSION or metadata.get("convention") != CONVENTION:
            raise ValueError("Incompatible model-server protocol or observation/action convention")

    def prepare_for_new_episode(self):
        self._request("/reset", {"protocol": VERSION, "session": self.session})
        self.active = True
        self.context = {}

    def set_inference_context(self, **context):
        self.context = context

    def predict(self, obs, task_description):
        if not self.active:
            self.prepare_for_new_episode()
        context, self.context = self.context, {}
        response = self._request("/predict", {
            "protocol": VERSION, "session": self.session,
            "observation": encode_observation(obs), "instruction": task_description, "context": context,
        })
        return list(validate_actions(response["actions"]))

    def close(self):
        try:
            if self.active:
                self._request("/close", {"protocol": VERSION, "session": self.session})
        finally:
            self.active = False
            self.url = None
