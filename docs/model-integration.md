# Model integration contract

`PolicyModel` defines `load(path)`, `predict(observation, instruction)`, `prepare_for_new_episode()`, `set_inference_context(**context)` and `close()`. `predict` returns a nonempty list/array of `[T, 7]` finite actions. The evaluator consumes each chunk in order, then requests a new chunk. Model state is reset for every independent rollout, including each adversarial placement candidate.

## Observations and actions

| Field | Contract |
|---|---|
| `image` | Main camera, RGB `uint8 [H,W,3]`; raw LIBERO image rotated 180° once |
| `wrist_image` | Wrist camera with the same convention |
| `state` | Eight floats: end-effector XYZ, XYZW quaternion converted to rotation vector, two gripper qpos values |
| `instruction` | The scene BDDL language instruction |
| return | `[T,7]` simulator OSC delta actions; last dimension already has LIBERO's gripper convention |

The environment returns native-resolution images (256×256 by default). Model-specific resizing happens **inside the model adapter process**:

- OpenPI: bilinear resize with padding to 224×224, plus its own server-side transforms. The actions returned by the OpenPI LIBERO policy already have the correct simulator convention.
- OpenVLA / VLA-Adapter / OFT: the original TensorFlow JPEG encode/decode and antialiased Lanczos3 resize to 224×224, followed by the upstream policy's crop/processor. Gripper normalization and inversion happen once in the upstream adapter.

The HTTP bridge transports canonical images and states without JPEG compression, flips, resizing or gripper conversion. Clients validate shape and finiteness. Values are not silently clipped into another action space. Custom robots/action spaces require a matching environment and policy adapter; this release targets the LIBERO Panda setup.

## HTTP service

`redvla serve` runs one loaded policy in its current interpreter. It defaults to `127.0.0.1:8001`. The service has:

- `GET /health`: readiness, protocol version and observation/action convention.
- `POST /reset`: reset policy state and acquire an episode lease.
- `POST /predict`: submit observation, instruction and one-shot inference context.
- `POST /close`: release the lease; the model remains loaded for the next evaluator.

Requests contain `protocol: 1` and a client session ID. NumPy arrays use dtype, shape and base64 raw bytes in JSON. Object dtypes and oversized requests are rejected; no pickle is used on the network. Request bodies are capped at 16 MiB. Inference requests are serialized. Competing sessions receive HTTP 409 rather than interleaving recurrent state. A lease expires after one hour of inactivity by default (`--lease-seconds`); normal client cleanup releases it immediately. A client timeout does not cancel an already executing GPU call.

For another machine, use an SSH tunnel or bind with `--host` on a trusted network. Setting `REDVLA_API_TOKEN` in both processes enables bearer authentication. Use TLS termination or an SSH tunnel when sending the token across an untrusted network; the built-in server itself is plain HTTP. Authentication values are read from environment variables and are not written into configs. This is a research inference service, not a multi-tenant public API.

One server per evaluation worker preserves episode isolation. Multiple endpoints may be routed by suite. Launch separate evaluator processes with disjoint task filters and output directories for parallel experiments. Sharing one stateful server across those processes is rejected.

## OpenPI service

The `openpi`, `pi0` and `pi05` adapter names select the same native OpenPI WebSocket client. It supports `ws://` and `wss://`, the metadata handshake and NumPy MessagePack payloads. Options are `replan_steps`, `connect_timeout`, `inference_timeout`, `image_size`, and `api_key_env`. `OPENPI_API_KEY`, when set, is sent using OpenPI's `Authorization: Api-Key ...` header.

Unlike a model hosted by `redvla serve`, the standard OpenPI server has no reset command. The client clears local inference context, and assumes the served OpenPI policy is stateless. Stateful custom OpenPI policies need their own reset protocol or a `PolicyModel` wrapper hosted by RedVLA. Attention context is only useful with a matching extended OpenPI server.

## External model repositories

`--source-root` explicitly chooses the upstream checkout containing `experiments/robot/robot_utils.py`. Environment-variable alternatives are `OPENVLA_ROOT`, `VLA_ADAPTER_ROOT`, and `OPENVLA_OFT_ROOT`. The model environment must already satisfy that checkout's dependencies, including the correct transformers/prismatic versions and optional FlashAttention build. RedVLA does not install conflicting model stacks in the evaluation environment.

Model options are passed through the adapter constructor. Choose the checkpoint's correct normalization key; for example, some VLA-Adapter checkpoints use `libero_goal_no_noops` while other policies use `libero_goal`. The server logs the selected key. Models trained per suite need their matching suite checkpoint, not just a different key.

Some upstream loaders update checkpoint configuration files during `get_model()`. Use a working checkpoint copy if the source weights/configuration must remain immutable. Their behavior is retained rather than globally monkey-patched.

## Adding another model

Implement a `PolicyModel` subclass in an importable module, then serve it from that model's environment:

```python
from redvla.models.base import PolicyModel

class MyPolicy(PolicyModel):
    def load(self, model_path):
        self.policy = load_my_policy(model_path)

    def predict(self, obs, task_description):
        return self.policy.predict(obs.image, obs.wrist_image, obs.state, task_description)
```

```bash
python -m redvla serve --model my_package.adapter:MyPolicy --checkpoint /path/to/weights
```

Import and implement `load_my_policy` in your module. Adapt the returned actions to the declared simulator convention. The evaluator then uses the standard `remote` adapter with no model-specific changes.
