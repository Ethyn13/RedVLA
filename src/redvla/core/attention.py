"""调度仿真推理请求并构造 openpi attention 参数。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import math

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from PIL import ImageDraw


def build_attention_output_root(
    *,
    episode_output_dir: str | Path,
    iteration: int,
    output_dir_name: str,
) -> Path:
    """为一次运行中的每个 task / episode / iteration 创建独立目录。"""
    episode_dir = Path(episode_output_dir)
    run_name = episode_dir.parents[1].name
    task_name = episode_dir.parent.name
    return (
        Path(__file__).resolve().parents[1]
        / output_dir_name
        / run_name
        / task_name
        / episode_dir.name
        / f"iteration_{int(iteration):03d}"
    )


def should_capture_attention(
    config: dict[str, Any] | None,
    *,
    query_index: int,
) -> bool:
    """判断当前第几次重规划推理是否需要提取 attention。"""
    if not config or not config.get("enabled", False):
        return False

    mode = str(config.get("query_mode", "first")).lower()
    if mode == "first":
        return query_index == 0
    if mode == "all":
        return True
    if mode == "stride":
        return query_index % int(config.get("query_stride", 1)) == 0
    return False


def build_attention_request(
    config: dict[str, Any],
    *,
    output_root: str | Path,
    timestep: int,
    query_index: int,
) -> dict[str, Any]:
    """构造 openpi Policy.infer 接受的 task_attention_request。"""
    return {
        "enabled": True,
        "output_root": str(output_root),
        "timestep": timestep,
        "query_index": query_index,
        "object_tokens": list(config["object_tokens"]),
        "render_action_attention": bool(config.get("render_action_attention", False)),
        "overlay_alpha": config.get("overlay_alpha", 0.45),
        "blur_radius": config.get("blur_radius", 2.0),
        "layer_scope": config.get("layer_scope", "all"),
        "head_agg": config.get("head_agg", "average"),
        "topk": config.get("topk", 8),
        "save_npy": bool(config.get("save_npy", True)),
    }


def finalize_attention_videos(
    *,
    output_root: str | Path,
    attention_names: list[str],
    total_steps: int,
    video_fps: int,
) -> None:
    """将重规划 attention 展开为逐仿真步的动态层级网格视频。"""
    for attention_name in attention_names:
        attention_root = Path(output_root) / attention_name
        overlays_root = attention_root / "overlays"
        layer_dirs = sorted(
            (
                path
                for path in overlays_root.iterdir()
                if path.is_dir() and path.name.isdigit()
            ),
            key=lambda path: int(path.name),
        ) if overlays_root.is_dir() else []
        if not layer_dirs:
            continue
        if attention_name == "action":
            for layer_dir in layer_dirs:
                for path in layer_dir.glob("step_*.png"):
                    timestep = int(path.stem.removeprefix("step_"))
                    if timestep >= int(total_steps):
                        path.unlink()

        timesteps = sorted(
            int(path.stem.removeprefix("step_"))
            for path in layer_dirs[0].glob("step_*.png")
        )
        if not timesteps:
            continue
        video_path = attention_root / "video" / "attention.mp4"
        temporary_path = attention_root / "video" / "attention.tmp.mp4"
        video_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(
            str(temporary_path),
            fps=int(video_fps),
            macro_block_size=1,
        )
        for step in range(int(total_steps)):
            source_timestep = timesteps[0]
            for timestep in timesteps:
                if timestep > step:
                    break
                source_timestep = timestep

            first = Image.open(
                layer_dirs[0] / f"step_{source_timestep:06d}.png"
            ).convert("RGB")
            width, height = first.size
            columns = min(6, len(layer_dirs))
            rows = int(math.ceil(len(layer_dirs) / columns))
            grid = Image.new("RGB", (width * columns, height * rows))
            for position, layer_dir in enumerate(layer_dirs):
                frame_path = (
                    layer_dir / f"step_{source_timestep:06d}.png"
                )
                if not frame_path.is_file():
                    continue
                tile = Image.open(frame_path).convert("RGB")
                draw = ImageDraw.Draw(tile)
                label = f"layer {layer_dir.name}"
                draw.rectangle((0, 0, 76, 20), fill=(0, 0, 0))
                draw.text((5, 4), label, fill=(255, 255, 255))
                row, column = divmod(position, columns)
                grid.paste(tile, (column * width, row * height))
            writer.append_data(np.asarray(grid))
        writer.close()
        temporary_path.replace(video_path)
