"""Data transforms for YAM robot policy.

Maps YAM camera names (top, left_wrist, right_wrist) to model image keys
(base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb) and handles action truncation.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms


def _parse_image(image) -> np.ndarray:
    """Parse image from LeRobot format (torch float32, C,H,W) to numpy uint8 (H,W,C)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


_FULL_CAMERA_MAP = {
    "top": "base_0_rgb",
    "left_wrist": "left_wrist_0_rgb",
    "right_wrist": "right_wrist_0_rgb",
}

_ARM_SLICES = {
    "left": slice(0, 7),
    "right": slice(7, 14),
}


@dataclasses.dataclass(frozen=True)
class YamInputs(transforms.DataTransformFn):
    """Inputs transform for YAM robot.

    Maps camera names to model-expected keys and passes through state/actions.

    Args:
        arm_id: ``"left"`` or ``"right"`` to select a single arm (7-dim
            state/action slice), or ``None`` for both arms (14-dim, default).
        cam_id: ``"left"`` for top + left_wrist, ``"right"`` for top +
            right_wrist, or ``None`` for all 3 cameras (default).
    """

    arm_id: str | None = None
    cam_id: str | None = None

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]

        # Select cameras based on cam_id.
        if self.cam_id is None:
            cam_keys = list(_FULL_CAMERA_MAP)
        elif self.cam_id == "left":
            cam_keys = ["top", "left_wrist"]
        elif self.cam_id == "right":
            cam_keys = ["top", "right_wrist"]
        else:
            raise ValueError(f"cam_id must be 'left', 'right', or None. Got: {self.cam_id!r}")

        images = {}
        image_masks = {}
        for source in cam_keys:
            dest = _FULL_CAMERA_MAP[source]
            if source in in_images:
                images[dest] = _parse_image(in_images[source])
                image_masks[dest] = np.True_
            else:
                raise ValueError(f"Expected camera '{source}' not found. Got: {tuple(in_images)}")

        # Slice state/actions for single-arm training.
        arm_slice = _ARM_SLICES.get(self.arm_id) if self.arm_id else None
        state = data["state"]
        if arm_slice is not None:
            state = state[..., arm_slice]

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        if "actions" in data:
            actions = np.asarray(data["actions"])
            if arm_slice is not None:
                actions = actions[..., arm_slice]
            inputs["actions"] = actions

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class YamOutputs(transforms.DataTransformFn):
    """Outputs transform for YAM robot.

    Truncates action output to the relevant dimensions.

    Args:
        arm_id: ``"left"`` or ``"right"`` to truncate to 7 dims (single arm),
            or ``None`` for 14 dims (both arms, default).
    """

    arm_id: str | None = None

    def __call__(self, data: dict) -> dict:
        n_dims = 7 if self.arm_id else 14
        return {"actions": np.asarray(data["actions"][:, :n_dims])}
