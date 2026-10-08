"""Local RoboCasa LeRobot-v2 datasets, without importing the simulator or LeRobot.

Language and modality layouts belong to each dataset root. The dataset returns
raw observations; RobocasaRepack is a dataset-only transform and is never applied
to policy inference inputs.
"""

from collections import OrderedDict
from collections.abc import Sequence
import dataclasses
import json
import logging
import operator
import os
from pathlib import Path

import cv2
import numpy as np
import polars as pl

STATE_ORDER = (
    "end_effector_position_relative",
    "end_effector_rotation_relative",
    "base_position",
    "base_rotation",
    "gripper_qpos",
)
ACTION_ORDER = (
    "end_effector_position",
    "end_effector_rotation",
    "gripper_close",
    "base_motion",
    "control_mode",
)
JOINT_STATE_ORDER = ("joint_pos", *STATE_ORDER)
JOINT_ACTION_ORDER = ("joint_pos", "gripper_close", "base_motion", "control_mode")
CAMERAS = {
    "observation/image": "robot0_agentview_left",
    "observation/wrist_image": "robot0_eye_in_hand",
    "observation/image_right": "robot0_agentview_right",
}


def _json(path: Path):
    with path.open() as handle:
        return json.load(handle)


def _jsonl(path: Path):
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _indices(modality: dict, kind: str, order: Sequence[str], size: int) -> np.ndarray:
    indices = []
    for component in order:
        if component not in modality[kind]:
            raise ValueError(f"Missing RoboCasa {kind} component {component!r} in modality.json")
        entry = modality[kind][component]
        if (
            entry.get("original_key", {"state": "observation.state", "action": "action"}[kind])
            != {"state": "observation.state", "action": "action"}[kind]
        ):
            raise ValueError(f"Unsupported source column for RoboCasa {kind}/{component}: {entry}")
        indices.extend(range(entry["start"], entry["end"]))
    if sorted(indices) != list(range(size)):
        raise ValueError(f"RoboCasa {kind} layout must cover {size} dimensions exactly once: {indices}")
    return np.asarray(indices)


@dataclasses.dataclass(frozen=True)
class RobocasaRepack:
    """Convert raw Groot ordering and camera keys into the policy input schema."""

    def __call__(self, data: dict) -> dict:
        state_indices, action_indices, camera_keys = data["_robocasa_layout"]
        return {
            **{target: data[source] for target, source in camera_keys.items()},
            "observation/state": np.asarray(data["observation.state"], dtype=np.float32)[..., state_indices],
            "actions": np.asarray(data["action"], dtype=np.float32)[..., action_indices],
            "prompt": data["prompt"],
        }


class RoboCasaDataset:
    """Frame-indexed, filtered local demos with action chunks clamped to their episode.

    Parquet tables and video handles are bounded per worker. Handles are opened
    lazily and reset after a process change, so spawned/forked workers do not share
    decoder state. Non-default mixture weights are rejected explicitly.
    """

    def __init__(
        self,
        data_dirs,
        action_horizon: int,
        *,
        prompt_from_task: bool,
        joint_control: bool = False,
        dataset_weights=None,
    ):
        if not prompt_from_task:
            raise ValueError("RoboCasa BC requires prompt_from_task=True; language comes from each demonstration")
        if action_horizon < 1:
            raise ValueError("action_horizon must be positive")
        if not data_dirs:
            raise ValueError("RoboCasa data_dirs must contain at least one local dataset")
        if dataset_weights is not None:
            raise ValueError("Weighted RoboCasa mixtures are not supported; omit dataset_weights for concatenation")
        self.action_horizon = action_horizon
        self.roots = []
        self.episodes = []
        self._episode_languages = {}
        self._ends = []
        self._tables = OrderedDict()
        self._videos = OrderedDict()
        self._pid = os.getpid()
        total = 0
        allowed_keys = {
            "path",
            "filter_key",
            "layout_and_style_ids",
            "fixture_refs",
            "object_categories",
            "obj_category",
            "episode_ids",
            "match_episode_id",
            "num_demos",
        }
        for entry in data_dirs:
            if unknown := set(entry) - allowed_keys:
                raise ValueError(f"Unsupported RoboCasa dataset settings: {sorted(unknown)}")
            if entry.get("filter_key") is not None:
                raise ValueError("RoboCasa filter_key is unsupported; use explicit episode_ids or num_demos")
            path = Path(entry["path"]).expanduser().resolve()
            info = _json(path / "meta/info.json")
            if not str(info.get("codebase_version", "")).startswith("v2"):
                raise ValueError(f"Expected a RoboCasa LeRobot-v2 dataset at {path}")
            modality = _json(path / "meta/modality.json")
            tasks = {int(task["task_index"]): task["task"] for task in _jsonl(path / "meta/tasks.jsonl")}
            camera_keys = {
                target: modality["video"][camera].get("original_key", f"observation.images.{camera}")
                for target, camera in CAMERAS.items()
                if camera in modality.get("video", {})
            }
            if not {"observation/image", "observation/wrist_image"}.issubset(camera_keys):
                raise ValueError(f"RoboCasa dataset requires left agentview and wrist cameras: {path}")
            sizes = (23, 13) if joint_control else (16, 12)
            layout = (
                _indices(modality, "state", JOINT_STATE_ORDER if joint_control else STATE_ORDER, sizes[0]),
                _indices(modality, "action", JOINT_ACTION_ORDER if joint_control else ACTION_ORDER, sizes[1]),
                camera_keys,
            )
            root = {"path": path, "info": info, "modality": modality, "tasks": tasks, "layout": layout}
            root_index = len(self.roots)
            self.roots.append(root)
            selected = []
            scenes = entry.get("layout_and_style_ids")
            allowed_scenes = set(map(tuple, scenes)) if scenes is not None else None
            explicit = entry.get("episode_ids")
            categories = entry.get("object_categories")
            if entry.get("obj_category") is not None:
                categories = [entry["obj_category"]] if categories is None else categories
            num_demos = entry.get("num_demos")
            if num_demos is not None and (num_demos == 0 or num_demos < -1):
                raise ValueError("num_demos must be positive, -1, or None")
            for episode in sorted(_jsonl(path / "meta/episodes.jsonl"), key=lambda item: item["episode_index"]):
                episode_id = int(episode["episode_index"])
                if explicit is not None and episode_id not in explicit:
                    continue
                if entry.get("match_episode_id") is not None and episode_id != entry["match_episode_id"]:
                    continue
                meta = _json(path / "extras" / f"episode_{episode_id:06d}" / "ep_meta.json")
                if allowed_scenes is not None and (meta["layout_id"], meta["style_id"]) not in allowed_scenes:
                    continue
                if any(
                    meta.get("fixture_refs", {}).get(key) != value
                    for key, value in (entry.get("fixture_refs") or {}).items()
                ):
                    continue
                objects = meta.get("object_cfgs", [])
                main = next((obj for obj in objects if obj.get("name") == "obj"), objects[0] if objects else {})
                category = main.get("info", {}).get("cat")
                if categories is not None and category not in categories:
                    continue
                if entry.get("obj_category") is not None and category != entry["obj_category"]:
                    continue
                self._episode_languages[(root_index, episode_id)] = meta.get("lang")
                selected.append((root_index, episode_id, int(episode["length"])))
                if num_demos is not None and num_demos > 0 and len(selected) == num_demos:
                    break
            if not selected:
                raise ValueError(f"No RoboCasa episodes match the filters at {path}: {entry}")
            logging.info("RoboCasa BC %s: episodes=%s", path, [ep[1] for ep in selected])
            for episode in selected:
                if episode[2] < 1:
                    raise ValueError(f"Empty RoboCasa episode: {episode}")
                self.episodes.append(episode)
                total += episode[2]
                self._ends.append(total)

    def __len__(self):
        return self._ends[-1]

    def __getstate__(self):
        return {**self.__dict__, "_videos": OrderedDict(), "_tables": OrderedDict()}

    def _table(self, root, episode_id, length):
        path = root["path"] / root["info"]["data_path"].format(
            episode_chunk=episode_id // root["info"]["chunks_size"],
            episode_index=episode_id,
        )
        if path not in self._tables:
            self._tables[path] = pl.read_parquet(path).to_dict(as_series=False)
            if len(self._tables) > 2:
                self._tables.popitem(last=False)
        self._tables.move_to_end(path)
        table = self._tables[path]
        if len(table["action"]) != length:
            raise ValueError(f"Episode length disagrees with metadata: {path}")
        return table

    def _frame(self, path, frame_index):
        if self._pid != os.getpid():
            self._videos = OrderedDict()
            self._pid = os.getpid()
        if path not in self._videos:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                raise ValueError(f"Cannot open RoboCasa video: {path}")
            self._videos[path] = capture
            if len(self._videos) > 6:
                self._videos.popitem(last=False)[1].release()
        self._videos.move_to_end(path)
        capture = self._videos[path]
        if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_index and not capture.set(
            cv2.CAP_PROP_POS_FRAMES, frame_index
        ):
            raise ValueError(f"Cannot seek RoboCasa video {path} to frame {frame_index}")
        ok, frame = capture.read()
        if not ok:
            raise ValueError(f"Missing frame {frame_index} in RoboCasa video {path}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        index = operator.index(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self._ends, index, side="right"))
        root_index, episode_id, length = self.episodes[position]
        frame_index = index - (self._ends[position - 1] if position else 0)
        root = self.roots[root_index]
        table = self._table(root, episode_id, length)
        annotation = root["modality"].get("annotation", {}).get("human.task_description", {})
        column = annotation.get("original_key", "annotation.human.task_description")
        if column not in table:
            column = "task_index"
        context = f"{root['path']}, episode={episode_id}, frame={frame_index}"
        if column not in table:
            raise ValueError(f"Missing natural-language task annotation: {context}")
        annotation_id = np.asarray(table[column][frame_index])
        if annotation_id.size != 1:
            raise ValueError(f"Expected one task-description ID: {context}")
        task_index = int(annotation_id.item())
        if "task_index" in table and task_index != int(np.asarray(table["task_index"][frame_index]).item()):
            raise ValueError(f"Task-description annotation disagrees with task_index: {context}")
        prompt = root["tasks"].get(task_index)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing or empty language for task_index={task_index}: {context}")
        recorded = self._episode_languages[(root_index, episode_id)]
        if isinstance(recorded, str) and recorded.strip() and prompt != recorded:
            raise ValueError(f"Task mapping disagrees with recorded ep_meta language: {context}")
        action_indices = np.minimum(np.arange(frame_index, frame_index + self.action_horizon), length - 1)
        sample = {
            "observation.state": np.asarray(table["observation.state"][frame_index], dtype=np.float32),
            "action": np.asarray([table["action"][i] for i in action_indices], dtype=np.float32),
            "prompt": prompt,
            "episode_index": episode_id,
            "frame_index": frame_index,
            "_robocasa_layout": root["layout"],
        }
        if sample["observation.state"].shape != root["layout"][0].shape or sample["action"].shape[-1] != len(
            root["layout"][1]
        ):
            raise ValueError(f"State/action dimensions disagree with modality metadata: {context}")
        for camera in root["layout"][2].values():
            path = root["path"] / root["info"]["video_path"].format(
                episode_chunk=episode_id // root["info"]["chunks_size"],
                episode_index=episode_id,
                video_key=camera,
            )
            sample[camera] = self._frame(path, frame_index)
        return sample

    def close(self):
        for capture in self._videos.values():
            capture.release()
        self._videos.clear()

    def __del__(self):
        if hasattr(self, "_videos"):
            self.close()
