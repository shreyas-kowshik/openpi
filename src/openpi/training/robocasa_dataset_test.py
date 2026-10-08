import dataclasses
import json
import pickle

import cv2
import numpy as np
import polars as pl
import pytest

from openpi import transforms
from openpi.models import pi0_config
from openpi.models import tokenizer
from openpi.training import config
from openpi.training import data_loader
from openpi.training import robocasa_dataset as rc


def _write_dataset(root, prompt="Pick the mug from the cabinet."):
    (root / "meta").mkdir(parents=True)
    modality = {
        "state": {},
        "action": {},
        "video": {},
        "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}},
    }
    for kind, keys, sizes, source in (
        (
            "state",
            (
                "base_position",
                "base_rotation",
                "end_effector_position_relative",
                "end_effector_rotation_relative",
                "gripper_qpos",
            ),
            (3, 4, 3, 4, 2),
            "observation.state",
        ),
        (
            "action",
            ("base_motion", "control_mode", "end_effector_position", "end_effector_rotation", "gripper_close"),
            (4, 1, 3, 3, 1),
            "action",
        ),
    ):
        start = 0
        for key, size in zip(keys, sizes, strict=True):
            modality[kind][key] = {"original_key": source, "start": start, "end": start + size}
            start += size
    cameras = list(rc.CAMERAS.values())
    for camera in cameras:
        modality["video"][camera] = {"original_key": f"observation.images.{camera}"}
    info = {
        "codebase_version": "v2.1",
        "fps": 20,
        "chunks_size": 1000,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/modality.json").write_text(json.dumps(modality))
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}) + "\n")
    episodes = []
    for ep_id, length in ((0, 4), (1, 3)):
        episodes.append({"episode_index": ep_id, "length": length})
        extras = root / "extras" / f"episode_{ep_id:06d}"
        extras.mkdir(parents=True)
        (extras / "ep_meta.json").write_text(
            json.dumps(
                {
                    "layout_id": ep_id + 1,
                    "style_id": 1,
                    "lang": prompt,
                    "fixture_refs": {"cab": f"cab_{ep_id}"},
                    "object_cfgs": [{"name": "obj", "info": {"cat": "mug"}}],
                }
            )
        )
        parquet = root / info["data_path"].format(episode_chunk=0, episode_index=ep_id)
        parquet.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame(
            {
                "observation.state": [list(np.arange(16, dtype=float) + ep_id * 100) for _ in range(length)],
                "action": [list(np.arange(12, dtype=float) + t * 10 + ep_id * 100) for t in range(length)],
                "annotation.human.task_description": [[0] for _ in range(length)],
                "task_index": [0] * length,
            }
        ).write_parquet(parquet)
        for camera in cameras:
            video = root / info["video_path"].format(
                episode_chunk=0,
                episode_index=ep_id,
                video_key=f"observation.images.{camera}",
            )
            video.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (16, 16))
            assert writer.isOpened()
            for t in range(length):
                writer.write(np.full((16, 16, 3), (ep_id + t) * 20, dtype=np.uint8))
            writer.release()
    (root / "meta/episodes.jsonl").write_text("".join(json.dumps(ep) + "\n" for ep in episodes))
    return root


@pytest.fixture
def dataset_root(tmp_path):
    return _write_dataset(tmp_path / "dataset")


def _dataset(root, **filters):
    return rc.RoboCasaDataset([{"path": str(root), **filters}], action_horizon=3, prompt_from_task=True)


def test_language_schema_and_episode_boundary(dataset_root):
    dataset = _dataset(dataset_root)
    sample = rc.RobocasaRepack()(dataset[3])
    assert sample["prompt"] == "Pick the mug from the cabinet."
    np.testing.assert_array_equal(sample["observation/state"], [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15])
    expected = np.asarray([5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4]) + 30
    np.testing.assert_array_equal(sample["actions"], np.tile(expected, (3, 1)))
    assert dataset[4]["episode_index"] == 1
    assert sample["observation/image_right"].shape == (16, 16, 3)
    dataset.close()


def test_filter_selection_is_deterministic(dataset_root):
    dataset = _dataset(
        dataset_root,
        layout_and_style_ids=[(2, 1)],
        fixture_refs={"cab": "cab_1"},
        object_categories=["mug"],
        episode_ids=[0, 1],
        num_demos=1,
    )
    assert dataset.episodes == [(0, 1, 3)]
    assert len(dataset) == 3
    assert _dataset(dataset_root, num_demos=1).episodes == [(0, 0, 4)]
    with pytest.raises(ValueError, match="No RoboCasa episodes"):
        _dataset(dataset_root, episode_ids=[99])


def test_task_ids_are_local_to_each_dataset(dataset_root, tmp_path):
    second = _write_dataset(tmp_path / "second", "Pick the banana from the fridge drawer.")
    dataset = rc.RoboCasaDataset([{"path": str(dataset_root)}, {"path": str(second)}], 2, prompt_from_task=True)
    assert dataset[0]["prompt"] == "Pick the mug from the cabinet."
    assert dataset[7]["prompt"] == "Pick the banana from the fridge drawer."
    dataset.close()


@pytest.mark.parametrize("prompt", ["", "   ", None])
def test_invalid_language_fails_before_inference(dataset_root, prompt):
    (dataset_root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": prompt}))
    with pytest.raises(ValueError, match="Missing or empty language.*episode=0"):
        _dataset(dataset_root)[0]


def test_missing_annotation_is_not_replaced_by_task_name(dataset_root):
    path = dataset_root / "data/chunk-000/episode_000000.parquet"
    frame = pl.read_parquet(path).drop("task_index", "annotation.human.task_description")
    frame.with_columns(pl.lit(0).alias("annotation.human.task_name")).write_parquet(path)
    with pytest.raises(ValueError, match="Missing natural-language task annotation"):
        _dataset(dataset_root)[0]


def test_wrong_task_mapping_does_not_silently_replace_recorded_language(dataset_root):
    (dataset_root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "PrepareCoffee"}))
    with pytest.raises(ValueError, match="Task mapping disagrees with recorded ep_meta language"):
        _dataset(dataset_root)[0]


def test_decoder_handles_are_not_pickled(dataset_root):
    dataset = _dataset(dataset_root)
    dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[0]["prompt"] == dataset[0]["prompt"]
    dataset.close()
    restored.close()


def test_unsupported_settings_fail_explicitly(dataset_root):
    with pytest.raises(ValueError, match="filter_key is unsupported"):
        _dataset(dataset_root, filter_key="1_demos")
    with pytest.raises(ValueError, match="Weighted RoboCasa mixtures"):
        rc.RoboCasaDataset([{"path": str(dataset_root)}], 2, prompt_from_task=True, dataset_weights=[1.0])


def test_bc_pipeline_passes_language_and_batches_once(dataset_root, tmp_path, monkeypatch):
    received = []

    class Tokenizer:
        def __init__(self, max_len):
            self.max_len = max_len

        def tokenize(self, text, state=None):
            received.append(text)
            return np.arange(self.max_len), np.ones(self.max_len, dtype=bool)

    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", Tokenizer)
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *_: None)
    model = pi0_config.Pi0Config(pi05=True, discrete_state_input=False, max_token_len=8, action_horizon=3)
    factory = config.LeRobotRobocasaDataConfig(data_dirs=[{"path": str(dataset_root)}])
    data_config = factory.create(tmp_path, model)
    assert data_config.prompt_from_task
    assert data_config.action_sequence_keys == ("action",)
    dataset = data_loader.create_torch_dataset(data_config, model.action_horizon, model)
    transformed = data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)
    loader = data_loader.TorchDataLoader(transformed, local_batch_size=2, num_batches=1, framework="pytorch")
    batch = next(iter(loader))
    assert batch["tokenized_prompt"].shape == (2, 8)
    assert batch["state"].shape == (2, 32)
    assert batch["actions"].shape == (2, 3, 32)
    assert received == ["Pick the mug from the cabinet."] * 2
    # The PyTorch and JAX loaders share the same numpy collation path.
    loader = data_loader.TorchDataLoader(transformed, local_batch_size=2, num_batches=1, framework="pytorch")
    assert next(iter(loader))["tokenized_prompt"].shape == (2, 8)
    dataset.close()


def test_joint_factory_inherits_language_and_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda *_: None)
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *_: None)
    factory = config.LeRobotRobocasaJointDataConfig(data_dirs=[{"path": "unused"}], episode_ids=[4])
    data_config = factory.create(tmp_path, pi0_config.Pi0Config(pi05=True))
    assert data_config.joint_control
    assert data_config.prompt_from_task
    assert data_config.data_dirs[0]["episode_ids"] == [4]


def test_joint_state_order_and_delta_actions(dataset_root, tmp_path, monkeypatch):
    modality_path = dataset_root / "meta/modality.json"
    modality = json.loads(modality_path.read_text())
    modality["state"]["joint_pos"] = {"original_key": "observation.state", "start": 16, "end": 23}
    modality["action"]["joint_pos"] = {"original_key": "action", "start": 5, "end": 12}
    modality["action"]["gripper_close"].update(start=12, end=13)
    modality_path.write_text(json.dumps(modality))
    path = dataset_root / "data/chunk-000/episode_000000.parquet"
    columns = pl.read_parquet(path).to_dict(as_series=False)
    columns["observation.state"] = [list(range(23))] * 4
    columns["action"] = [list(range(13))] * 4
    pl.DataFrame(columns).write_parquet(path)
    dataset = rc.RoboCasaDataset(
        [{"path": str(dataset_root), "episode_ids": [0]}],
        2,
        prompt_from_task=True,
        joint_control=True,
    )
    raw = dataset[0]
    monkeypatch.setattr(tokenizer, "PaligemmaTokenizer", lambda *_: None)
    monkeypatch.setattr(config.DataConfigFactory, "_load_norm_stats", lambda *_: None)
    factory = config.LeRobotRobocasaJointDataConfig(data_dirs=[{"path": str(dataset_root)}])
    data_config = factory.create(tmp_path, pi0_config.Pi0Config(pi05=True))
    sample = transforms.compose([*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs])(raw)
    np.testing.assert_array_equal(sample["state"][:7], np.arange(16, 23))
    np.testing.assert_array_equal(sample["actions"][:, :7], np.full((2, 7), -11))
    output = transforms.compose(data_config.data_transforms.outputs)(sample)
    np.testing.assert_array_equal(output["actions"][:, :7], np.tile(np.arange(5, 12), (2, 1)))
    assert output["actions"].shape == (2, 13)
    dataset.close()


def test_normalization_loader_keeps_final_partial_batch():
    dataset = [{"state": np.asarray([index], dtype=np.float32)} for index in range(3)]
    loader = data_loader.TorchDataLoader(
        dataset,
        local_batch_size=2,
        num_batches=2,
        framework="pytorch",
        drop_last=False,
    )
    batches = list(loader)
    assert [batch["state"].shape[0] for batch in batches] == [2, 1]


@pytest.mark.parametrize(
    ("name", "legacy_name"),
    [
        (
            "pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32",
            "pi05_robocasa_single_task_lora_exact_replay_l1_s1_ep32",
        ),
        (
            "pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37",
            "pi05_robocasa_single_task_lora_vision_fullft_action_pick_place_fridge_drawer_to_shelf_l50_s37",
        ),
        (
            "pi05_robocasa_bc_prepare_coffee_l25_s29",
            "pi05_robocasa_single_task_lora_vision_fullft_action_prepare_coffee_l25_s29",
        ),
    ],
)
def test_standalone_bc_configs_preserve_training_settings_and_separate_legacy_assets(name, legacy_name):
    new, legacy = config.get_config(name), config.get_config(legacy_name)
    assert not hasattr(config, "ROBOCASA_BC_SOURCES")
    assert new.data.assets.asset_id == name
    assert legacy.data.assets.asset_id == "robocasa"
    assert new.model.discrete_state_input is False
    assert new.model == dataclasses.replace(legacy.model, discrete_state_input=False)
    assert new.model is not legacy.model
    assert new.data is not legacy.data
    assert new.data.data_dirs is not legacy.data.data_dirs
    assert new.data.data_dirs[0] is not legacy.data.data_dirs[0]
    for field in (
        "data_dirs",
        "dataset_weights",
        "action_dim",
        "layout_and_style_ids",
        "fixture_refs",
        "episode_ids",
        "num_demos",
        "match_episode_id",
        "object_categories",
        "obj_category",
    ):
        assert getattr(new.data, field) == getattr(legacy.data, field), field
    for field in (
        "batch_size",
        "num_train_steps",
        "num_workers",
        "save_interval",
        "keep_period",
        "log_interval",
        "action_l1_loss_interval",
        "action_dim",
        "ema_decay",
        "lr_schedule",
        "optimizer",
    ):
        assert getattr(new, field) == getattr(legacy, field), field
    assert new.seed == 0
    assert new.project_name == "robocasa-openpi-bc"
    assert new.checkpoint_base_dir.endswith("/robocasa_bc")
    assert new.checkpoint_base_dir != legacy.checkpoint_base_dir
    assert new.assets_dirs != legacy.assets_dirs
    assert new.data.eval_dataset_path == new.data.data_dirs[0]["path"]
    assert new.data.eval_init_mode == "exact_state_replay"
    expected_episode = {
        "pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32": 32,
        "pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37": 40,
        "pi05_robocasa_bc_prepare_coffee_l25_s29": 4,
    }[name]
    assert new.data.eval_pool_episode_ids == [expected_episode]
    assert new.data.eval_keep_robot_pose is True
    assert new.data.eval_robot_pose_noise == 0.0
    assert new.data.eval_object_pose_noise == 0.0
    assert new.data.eval_object_ori_noise == 0.0
