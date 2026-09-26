"""OpenVLA semantic text-to-image attention rendering.

The output layout intentionally matches adversarial's Pi0/Pi05 visualizer:
``<output_root>/<target>/overlays/<layer>/step_<timestep>.png``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, ImageFilter


def resolve_lm_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    language_model = getattr(model, "language_model", None)
    candidates = (
        getattr(language_model, "model", None),
        language_model,
        getattr(language_model, "transformer", None),
        model,
        getattr(model, "model", None),
    )
    for candidate in candidates:
        if candidate is None:
            continue
        for name in ("layers", "h", "blocks"):
            layers = getattr(candidate, name, None)
            if isinstance(layers, (torch.nn.ModuleList, list, tuple)):
                return layers
    raise ValueError("无法定位 OpenVLA language model layers")


def resolve_num_image_patches(model: torch.nn.Module) -> int:
    vision_backbone = getattr(model, "vision_backbone", None)
    if vision_backbone is None:
        raise ValueError("OpenVLA 模型缺少 vision_backbone")
    for name in ("get_num_patches", "num_patches"):
        value = getattr(vision_backbone, name, None)
        if value is not None:
            return int(value() if callable(value) else value)
    for name in (
        "featurizer",
        "dino_featurizer",
        "clip_featurizer",
        "siglip_featurizer",
    ):
        featurizer = getattr(vision_backbone, name, None)
        patch_embed = getattr(featurizer, "patch_embed", None)
        value = getattr(patch_embed, "num_patches", None)
        if value is not None:
            return int(value)
    raise ValueError("无法确定 OpenVLA 图像 patch 数")


def parse_layer_scope(scope: str, layer_count: int) -> list[int]:
    available = list(range(int(layer_count)))
    text = str(scope or "all").strip().lower()
    if text == "all":
        return available
    if text == "last":
        return available[-1:]
    if text == "last4":
        return available[-4:]

    selected: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        match = re.fullmatch(r"(-?\d+)\s*[:-]\s*(-?\d+)", part)
        raw_indices: Sequence[int]
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            start = start if start >= 0 else layer_count + start
            end = end if end >= 0 else layer_count + end
            step = 1 if start <= end else -1
            raw_indices = range(start, end + step, step)
        else:
            index = int(part)
            raw_indices = [index if index >= 0 else layer_count + index]
        for index in raw_indices:
            if not 0 <= index < layer_count:
                raise ValueError(
                    f"OpenVLA attention layer {index} 超出范围 "
                    f"[0, {layer_count - 1}]"
                )
            if index not in selected:
                selected.append(index)
    if not selected:
        raise ValueError(f"无效的 OpenVLA layer_scope: {scope!r}")
    return selected


def _token_ids(tokenized: Any) -> list[int]:
    if isinstance(tokenized, dict):
        values = tokenized["input_ids"]
    else:
        values = tokenized.input_ids
    if torch.is_tensor(values):
        values = values.detach().cpu().tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def find_phrase_token_indices(
    tokenizer: Any,
    input_ids: torch.Tensor,
    phrase: str,
) -> list[int]:
    """Find the last prompt occurrence of an object phrase by token IDs."""
    prompt_ids = _token_ids({"input_ids": input_ids})
    normalized = str(phrase).replace("_", " ").strip()
    candidates: list[list[int]] = []
    for text in (f" {normalized}", normalized):
        try:
            ids = _token_ids(
                tokenizer(text, add_special_tokens=False)
            )
        except (KeyError, TypeError):
            continue
        if ids and ids not in candidates:
            candidates.append(ids)

    best: list[int] | None = None
    for candidate in candidates:
        width = len(candidate)
        for start in range(len(prompt_ids) - width + 1):
            if prompt_ids[start : start + width] == candidate:
                best = list(range(start, start + width))
    if best is None:
        raise ValueError(
            f"任务提示词中没有找到目标 token: {phrase!r} "
            f"(normalized={normalized!r})"
        )
    return best


def _extract_attention_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output) and output.ndim == 4:
        return output
    if isinstance(output, (tuple, list)):
        for value in output:
            attention = _extract_attention_tensor(value)
            if attention is not None:
                return attention
    return None


def _aggregate_heads(
    per_head: np.ndarray,
    method: str,
    topk: int,
) -> np.ndarray:
    if method == "average":
        return per_head.mean(axis=0)
    if method != "topk":
        raise ValueError("head_agg 必须是 average 或 topk")
    count = min(max(1, int(topk)), per_head.shape[0])
    head_scores = per_head.mean(axis=1)
    indices = np.argpartition(head_scores, -count)[-count:]
    return per_head[indices].mean(axis=0)


def _infer_grid(num_patches: int) -> tuple[int, int]:
    side = int(round(math.sqrt(num_patches)))
    if side * side == num_patches:
        return side, side
    for height in range(side, 0, -1):
        if num_patches % height == 0:
            return height, num_patches // height
    raise ValueError(f"无法把 {num_patches} 个 patch 排成二维网格")


def _jet(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def _render_overlay(
    image_rgb: np.ndarray,
    vector: np.ndarray,
    overlay_alpha: float,
    blur_radius: float,
) -> Image.Image:
    height, width = _infer_grid(vector.size)
    heatmap = vector.reshape(height, width)
    minimum, maximum = float(heatmap.min()), float(heatmap.max())
    heatmap = (heatmap - minimum) / max(maximum - minimum, 1e-8)
    base = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    colored = Image.fromarray(
        (_jet(heatmap) * 255).astype(np.uint8)
    ).resize(base.size, Image.Resampling.BILINEAR)
    if float(blur_radius) > 0:
        colored = colored.filter(
            ImageFilter.GaussianBlur(float(blur_radius))
        )
    return Image.blend(base, colored, float(overlay_alpha))


def _write_layer_output(
    *,
    output_root: Path,
    target_token: str,
    layer_idx: int,
    timestep: int,
    vector: np.ndarray,
    image_rgb: np.ndarray,
    prompt_token_indices: list[int],
    num_image_patches: int,
    query_index: int,
    head_agg: str,
    topk: int,
    overlay_alpha: float,
    blur_radius: float,
    save_npy: bool,
) -> None:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_token).strip("_")
    stem = f"step_{int(timestep):06d}"
    target_root = output_root / name
    overlay_dir = target_root / "overlays" / str(layer_idx)
    meta_dir = target_root / "meta" / str(layer_idx)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    overlay = _render_overlay(
        image_rgb, vector, overlay_alpha, blur_radius
    )
    overlay_path = overlay_dir / f"{stem}.png"
    overlay.save(overlay_path)
    if save_npy:
        npy_dir = target_root / "raw_npy" / str(layer_idx)
        npy_dir.mkdir(parents=True, exist_ok=True)
        np.save(npy_dir / f"{stem}.npy", vector.astype(np.float32))

    metadata = {
        "target_token": target_token,
        "layer": int(layer_idx),
        "timestep": int(timestep),
        "query_index": int(query_index),
        "prompt_token_indices": prompt_token_indices,
        "query_token_indices": [
            int(num_image_patches) + index
            for index in prompt_token_indices
        ],
        "image_key_range": [1, 1 + int(num_image_patches)],
        "num_image_patches": int(num_image_patches),
        "head_aggregation": head_agg,
        "topk": int(topk),
        "overlay_alpha": float(overlay_alpha),
        "blur_radius": float(blur_radius),
        "overlay_path": str(overlay_path),
    }
    (meta_dir / f"{stem}.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def capture_openvla_action_attention(
    *,
    model: torch.nn.Module,
    tokenizer: Any,
    model_inputs: dict[str, Any],
    image_rgb: np.ndarray,
    object_tokens: list[str],
    output_root: str | Path,
    timestep: int,
    query_index: int,
    unnorm_key: str,
    layer_scope: str = "all",
    head_agg: str = "average",
    topk: int = 8,
    overlay_alpha: float = 0.45,
    blur_radius: float = 2.0,
    save_npy: bool = True,
    strict: bool = True,
) -> np.ndarray:
    """Predict one action and render each semantic target's image attention."""
    input_ids = model_inputs["input_ids"]
    num_patches = resolve_num_image_patches(model)
    target_positions: dict[str, list[int]] = {}
    for target in object_tokens:
        try:
            target_positions[target] = find_phrase_token_indices(
                tokenizer, input_ids, target
            )
        except ValueError:
            if strict:
                raise

    layers = resolve_lm_layers(model)
    selected_layers = parse_layer_scope(layer_scope, len(layers))
    captured: dict[int, dict[str, np.ndarray]] = {}
    handles: list[Any] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            attention = _extract_attention_tensor(output)
            if attention is None or attention.ndim != 4:
                return
            image_start, image_end = 1, 1 + num_patches
            per_target: dict[str, np.ndarray] = {}
            for target, prompt_indices in target_positions.items():
                query_indices = [
                    num_patches + index for index in prompt_indices
                ]
                if (
                    max(query_indices, default=-1) >= attention.shape[-2]
                    or image_end > attention.shape[-1]
                ):
                    continue
                per_head = (
                    attention[0, :, query_indices, image_start:image_end]
                    .mean(dim=1)
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                )
                per_target[target] = _aggregate_heads(
                    per_head, head_agg, topk
                )
            if per_target:
                captured[layer_idx] = per_target

        return hook

    for layer_idx in selected_layers:
        attention_module = None
        for name in ("self_attn", "attention", "attn"):
            attention_module = getattr(layers[layer_idx], name, None)
            if attention_module is not None:
                break
        if attention_module is None:
            if strict:
                raise ValueError(
                    f"OpenVLA layer {layer_idx} 没有 attention module"
                )
            continue
        handles.append(
            attention_module.register_forward_hook(make_hook(layer_idx))
        )

    try:
        action = model.predict_action(
            **model_inputs,
            unnorm_key=unnorm_key,
            do_sample=False,
            output_attentions=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    if strict and not captured:
        raise RuntimeError("OpenVLA 未返回可提取的 eager attention")

    root = Path(output_root)
    for layer_idx, per_target in captured.items():
        for target, vector in per_target.items():
            _write_layer_output(
                output_root=root,
                target_token=target,
                layer_idx=layer_idx,
                timestep=timestep,
                vector=vector,
                image_rgb=image_rgb,
                prompt_token_indices=target_positions[target],
                num_image_patches=num_patches,
                query_index=query_index,
                head_agg=head_agg,
                topk=topk,
                overlay_alpha=overlay_alpha,
                blur_radius=blur_radius,
                save_npy=save_npy,
            )
    return action
