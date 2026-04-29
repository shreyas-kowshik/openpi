"""
Visualize robomimic tool_hang demo episodes as individual .mp4 files.

Loads the raw HDF5 dataset, replays each episode's MuJoCo states in the
robosuite simulator, and renders agentview + wrist camera to separate videos.

Output: robomimic_data_vis/tool_hang_ep_<idx>.mp4
"""

import os
import argparse
import h5py
import imageio
import numpy as np

import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils


def render_episode(env, initial_state, states, video_path, camera_names, video_skip=5, fps=20):
    """Render a single episode to an mp4 file."""
    writer = imageio.get_writer(video_path, fps=fps)
    env.reset_to(initial_state)

    for i in range(states.shape[0]):
        env.reset_to({"states": states[i]})
        if i % video_skip == 0:
            frames = []
            for cam in camera_names:
                frames.append(env.render(mode="rgb_array", height=512, width=512, camera_name=cam))
            frame = np.concatenate(frames, axis=1)
            writer.append_data(frame)

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="/data/hf_cache/datasets/robomimic/tool_hang/ph/demo_v15.hdf5",
                        help="Path to raw HDF5 dataset")
    parser.add_argument("--output_dir", type=str,
                        default="robomimic_data_vis",
                        help="Directory for output mp4 files")
    parser.add_argument("--camera_names", type=str, nargs="+",
                        default=["agentview", "robot0_eye_in_hand"],
                        help="Camera names to render (concatenated horizontally)")
    parser.add_argument("--video_skip", type=int, default=5,
                        help="Write every N-th frame")
    parser.add_argument("--fps", type=int, default=20,
                        help="Video FPS")
    parser.add_argument("--n", type=int, default=None,
                        help="Limit to first N episodes (default: all)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize obs utils (required by robomimic)
    dummy_spec = dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)

    # Create environment from dataset metadata
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    f = h5py.File(args.dataset, "r")
    demos = sorted(list(f["data"].keys()), key=lambda x: int(x.split("_")[1]))

    if args.n is not None:
        demos = demos[:args.n]

    print(f"Rendering {len(demos)} episodes to {args.output_dir}/")

    for idx, ep in enumerate(demos):
        ep_idx = int(ep.split("_")[1])
        video_path = os.path.join(args.output_dir, f"tool_hang_ep_{ep_idx}.mp4")

        states = f[f"data/{ep}/states"][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f[f"data/{ep}"].attrs["model_file"]
            initial_state["ep_meta"] = f[f"data/{ep}"].attrs.get("ep_meta", None)

        print(f"  [{idx+1}/{len(demos)}] {ep} ({states.shape[0]} steps) -> {video_path}")
        render_episode(env, initial_state, states, video_path,
                       camera_names=args.camera_names,
                       video_skip=args.video_skip,
                       fps=args.fps)

    f.close()
    print(f"Done. {len(demos)} videos saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
