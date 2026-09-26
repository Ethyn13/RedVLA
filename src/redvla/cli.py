"""Public command-line interface; help, config validation and serving are simulator-free."""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="redvla", description="Physical red teaming for VLA policies")
    parser.add_argument("--version", action="version", version="redvla 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("models", help="List built-in model adapters")
    commands.add_parser("doctor", help="Report dependency availability without loading any models")
    for name, description in [("evaluate", "Evaluate fixed scenes with safety rules"),
                               ("attack", "Run iterative adversarial placement search"),
                               ("benign", "Evaluate task completion without safety monitoring"),
                               ("validate", "Validate and expand configuration without importing simulator")]:
        command = commands.add_parser(name, help=description)
        command.add_argument("--config", required=True)
        command.add_argument("--task", help="Match task category or scene path substring")
        command.add_argument("--limit", type=int)
        command.add_argument("--max-steps", type=int)
        command.add_argument("--output-dir")
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--gpu", help="Set CUDA_VISIBLE_DEVICES before importing simulation/model code")
    serve = commands.add_parser("serve", help="Serve a model from its own Python environment")
    serve.add_argument("--model", required=True)
    serve.add_argument("--checkpoint", default="")
    serve.add_argument("--source-root")
    serve.add_argument("--options", default="{}", help="JSON model constructor options")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--token-env", default="REDVLA_API_TOKEN")
    serve.add_argument("--lease-seconds", type=float, default=3600)
    serve.add_argument("--gpu")
    probe = commands.add_parser("probe", help="Test a policy endpoint with one observation")
    probe.add_argument("--model", choices=["remote", "openpi", "pi0", "pi05"], required=True)
    probe.add_argument("--endpoint", default="")
    probe.add_argument("--observation", help="NPZ containing image, wrist_image, state; otherwise use synthetic input")
    probe.add_argument("--instruction", default="pick up the bowl")
    probe.add_argument("--options", default="{}")
    args = parser.parse_args(argv)
    try:
        if getattr(args, "gpu", None) is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if args.command == "models":
            from redvla.models.factory import list_models
            print("\n".join(list_models()))
        elif args.command == "doctor":
            from redvla.envs.backend import backend_report
            print(json.dumps({"python": sys.executable, "version": sys.version,
                              "simulator": backend_report(),
                              "dependencies": {name: importlib.util.find_spec(name) is not None
                               for name in ("numpy", "yaml", "PIL", "msgpack", "websockets", "torch",
                                            "mujoco", "robosuite", "libero", "bddl", "tensorflow")}}, indent=2))
        elif args.command == "serve":
            from redvla.models.factory import create_model
            from redvla.serving import make_server
            if args.lease_seconds <= 0:
                raise ValueError("lease-seconds must be positive")
            options = json.loads(args.options)
            if not isinstance(options, dict):
                raise ValueError("options must be a JSON object")
            model = create_model(args.model, args.checkpoint, options, args.source_root)
            try:
                with make_server(model, args.host, args.port, os.environ.get(args.token_env), args.lease_seconds) as server:
                    print(f"READY http://{args.host}:{server.server_port} model={args.model}", flush=True)
                    server.serve_forever()
            finally:
                model.close()
        elif args.command == "probe":
            import numpy as np
            from redvla.core.types import Observation
            from redvla.models.factory import create_model
            if args.observation:
                with np.load(args.observation, allow_pickle=False) as data:
                    obs = Observation(data["image"], data["wrist_image"], data["state"])
            else:
                obs = Observation(np.zeros((256, 256, 3), np.uint8),
                                  np.zeros((256, 256, 3), np.uint8), np.zeros(8, np.float32))
            model = create_model(args.model, args.endpoint, json.loads(args.options))
            try:
                model.prepare_for_new_episode()
                actions = np.asarray(model.predict(obs, args.instruction))
                print(json.dumps({"shape": list(actions.shape), "finite": bool(np.isfinite(actions).all()),
                                  "synthetic_observation": args.observation is None,
                                  "first_action": actions[0].tolist()}))
            finally:
                model.close()
        else:
            from redvla.config import load_config, serialize_jobs
            mode = "evaluate" if args.command == "validate" else args.command
            output, jobs = load_config(args.config, mode, args.task, args.limit, args.max_steps)
            if args.dry_run or args.command == "validate":
                print(json.dumps({"output_dir": args.output_dir or output, "job_count": len(jobs),
                                  "jobs": serialize_jobs(jobs)}, indent=2))
            else:
                from redvla.evaluation import run_jobs
                run_jobs(str(Path(args.output_dir).resolve()) if args.output_dir else output, jobs, mode)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (ValueError, FileNotFoundError, ImportError, TimeoutError) as error:
        parser.exit(2, f"redvla: {error}\n")


if __name__ == "__main__":
    main()
