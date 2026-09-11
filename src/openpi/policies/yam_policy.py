"""Data transforms for the 14-DoF YAM bimanual robot."""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    value = np.asarray(image)
    if np.issubdtype(value.dtype, np.floating):
        value = (255.0 * value).clip(0, 255).astype(np.uint8)
    if value.ndim == 3 and value.shape[0] == 3:
        value = einops.rearrange(value, "c h w -> h w c")
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError(f"Expected a three-channel YAM image, got {value.shape}")
    return np.ascontiguousarray(value, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    camera_map: ClassVar[dict[str, str]] = {
        "top": "base_0_rgb",
        "left_wrist": "left_wrist_0_rgb",
        "right_wrist": "right_wrist_0_rgb",
    }

    def __call__(self, data: dict) -> dict:
        source = data["images"]
        missing = [camera for camera in self.camera_map if camera not in source]
        if missing:
            raise ValueError(f"Missing YAM cameras {missing}; received {tuple(source)}")
        output = {
            "image": {target: _parse_image(source[camera]) for camera, target in self.camera_map.items()},
            "image_mask": dict.fromkeys(self.camera_map.values(), np.True_),
            "state": np.asarray(data["state"]),
        }
        if "actions" in data:
            output["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            output["prompt"] = data["prompt"]
        return output


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., :14]}


__all__ = ["YamInputs", "YamOutputs"]
