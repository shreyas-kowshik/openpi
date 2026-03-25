"""Collect 5 successful demonstrations for the KITCHEN_SCENE8 task with use_swap.

Uses proportional control (gain=25) with simultaneous XYZ tracking.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from teleop_scratch.env_helper import (
    create_env,
    reset_episode,
    get_state,
    render_obs,
    render_obs_raw,
)
from teleop_scratch.save_demo import save_demo, save_demos_combined, verify_hdf5


def _get_body_pos(env, body_name):
    bid = env.sim.model.body_name2id(body_name)
    return env.sim.data.body_xpos[bid].copy()


def _move_to(env, target, obs, traj, gripper=-1.0, max_steps=150,
             threshold=0.008, gain=25.0, step_counter=0, verbose=True):
    """Move EEF to target using proportional control (simultaneous XYZ)."""
    target = np.array(target, dtype=np.float64)
    for i in range(max_steps):
        eef = get_state(obs)["eef_pos"]
        err = target - eef
        dist = np.linalg.norm(err)
        if dist < threshold:
            if verbose and i > 0:
                print(f"    converged in {i} steps, dist={dist:.4f}")
            break
        action_xyz = np.clip(err * gain, -1.0, 1.0)
        action = np.array([action_xyz[0], action_xyz[1], action_xyz[2], 0, 0, 0, gripper])
        obs, reward, done, info = env.step(action.tolist())
        step_counter += 1
        _record(obs, action, traj)
        if done:
            return obs, step_counter, True
    else:
        if verbose:
            eef = get_state(obs)["eef_pos"]
            print(f"    max_steps={max_steps}, final dist={np.linalg.norm(target - eef):.4f}")
    return obs, step_counter, False


def _gripper_action(env, obs, traj, gripper, num_steps=15, step_counter=0):
    """Open or close gripper."""
    for i in range(num_steps):
        action = np.array([0, 0, 0, 0, 0, 0, gripper])
        obs, reward, done, info = env.step(action.tolist())
        step_counter += 1
        _record(obs, action, traj)
        if done:
            return obs, step_counter, True
    return obs, step_counter, False


def _record(obs, action, traj):
    state = get_state(obs)
    base_img, wrist_img = render_obs_raw(obs)
    traj["images_base"].append(base_img)
    traj["images_wrist"].append(wrist_img)
    traj["ee_states"].append(
        np.concatenate([state["eef_pos"], state["eef_axisangle"]]).astype(np.float32)
    )
    traj["gripper_states"].append(state["gripper"].astype(np.float32))
    traj["actions"].append(action.astype(np.float32).copy())


def _init_traj(obs):
    state = get_state(obs)
    base_img, wrist_img = render_obs_raw(obs)
    return {
        "images_base": [base_img],
        "images_wrist": [wrist_img],
        "ee_states": [np.concatenate([state["eef_pos"], state["eef_axisangle"]]).astype(np.float32)],
        "gripper_states": [state["gripper"].astype(np.float32)],
        "actions": [],
        "success": False,
        "total_steps": 0,
    }


def pick_and_place(env, obs, traj, pot_name, place_target, step_counter=0):
    """Pick a pot and place it at place_target."""
    pot_pos = _get_body_pos(env, pot_name)
    print(f"\n  Picking {pot_name} at {np.round(pot_pos, 4)}")

    # 1) Move above pot (high approach)
    above = [pot_pos[0], pot_pos[1], 1.10]
    print(f"  [1] Above pot")
    obs, step_counter, done = _move_to(env, above, obs, traj, gripper=-1.0,
                                        max_steps=120, threshold=0.012, step_counter=step_counter)
    if done: return obs, step_counter, done

    # 2) Lower to grasp (simultaneous XYZ)
    pot_pos = _get_body_pos(env, pot_name)  # re-read
    grasp = [pot_pos[0], pot_pos[1], pot_pos[2] - 0.01]
    print(f"  [2] Lowering to grasp at z={grasp[2]:.3f}")
    obs, step_counter, done = _move_to(env, grasp, obs, traj, gripper=-1.0,
                                        max_steps=100, threshold=0.006, step_counter=step_counter)
    if done: return obs, step_counter, done

    # 3) Close gripper
    print(f"  [3] Closing gripper")
    obs, step_counter, done = _gripper_action(env, obs, traj, gripper=1.0,
                                               num_steps=20, step_counter=step_counter)
    if done: return obs, step_counter, done

    # 4) Lift up
    eef = get_state(obs)["eef_pos"]
    lift = [eef[0], eef[1], 1.10]
    print(f"  [4] Lifting")
    obs, step_counter, done = _move_to(env, lift, obs, traj, gripper=1.0,
                                        max_steps=80, threshold=0.012, step_counter=step_counter)
    if done: return obs, step_counter, done

    # Check if pot was lifted
    pot_after = _get_body_pos(env, pot_name)
    lifted = pot_after[2] > (pot_pos[2] + 0.02)
    print(f"    pot z: {pot_pos[2]:.3f} -> {pot_after[2]:.3f}  lifted={lifted}")

    # 5) Move above place target
    above_place = [place_target[0], place_target[1], 1.10]
    print(f"  [5] Above stove target")
    obs, step_counter, done = _move_to(env, above_place, obs, traj, gripper=1.0,
                                        max_steps=150, threshold=0.012, step_counter=step_counter)
    if done: return obs, step_counter, done

    # 6) Lower to place
    place = [place_target[0], place_target[1], 0.96]
    print(f"  [6] Placing at {np.round(place, 3)}")
    obs, step_counter, done = _move_to(env, place, obs, traj, gripper=1.0,
                                        max_steps=80, threshold=0.008, step_counter=step_counter)
    if done: return obs, step_counter, done

    # 7) Open gripper
    print(f"  [7] Opening gripper")
    obs, step_counter, done = _gripper_action(env, obs, traj, gripper=-1.0,
                                               num_steps=15, step_counter=step_counter)
    if done: return obs, step_counter, done

    # 8) Retreat up
    eef = get_state(obs)["eef_pos"]
    retreat = [eef[0], eef[1], 1.10]
    print(f"  [8] Retreating up")
    obs, step_counter, done = _move_to(env, retreat, obs, traj, gripper=-1.0,
                                        max_steps=60, threshold=0.015, step_counter=step_counter)

    pot_final = _get_body_pos(env, pot_name)
    print(f"    {pot_name} final: {np.round(pot_final, 4)}")
    print(f"    check_success: {env.check_success()}")

    return obs, step_counter, done


def collect_one_demo(env, init_states, init_idx, demo_idx,
                     output_dir="teleop_scratch/renders_collect"):
    print(f"\n{'='*60}")
    print(f"DEMO {demo_idx} (init_state={init_idx})")
    print(f"{'='*60}")

    obs = reset_episode(env, init_states, idx=init_idx)
    traj = _init_traj(obs)

    pot1 = _get_body_pos(env, "moka_pot_1_main")
    pot2 = _get_body_pos(env, "moka_pot_2_main")
    stove = _get_body_pos(env, "flat_stove_1_main")
    burner = _get_body_pos(env, "flat_stove_1_burner")
    eef = get_state(obs)["eef_pos"]

    print(f"EEF:    {np.round(eef, 4)}")
    print(f"Pot1:   {np.round(pot1, 4)}")
    print(f"Pot2:   {np.round(pot2, 4)}")
    print(f"Stove:  {np.round(stove, 4)}")
    print(f"Burner: {np.round(burner, 4)}")

    # cook_region is on the stove — target is between stove center and burner
    stove_center_x = (stove[0] + burner[0]) / 2
    stove_y = stove[1]

    # Two placement positions (offset in x so both pots fit)
    target_1 = np.array([stove_center_x - 0.02, stove_y, stove[2]])
    target_2 = np.array([stove_center_x + 0.04, stove_y, stove[2]])

    # Pick farther pot first to avoid knocking the closer one
    d1 = np.linalg.norm(pot1[:2] - np.array([stove_center_x, stove_y]))
    d2 = np.linalg.norm(pot2[:2] - np.array([stove_center_x, stove_y]))

    if d1 >= d2:
        order = [("moka_pot_1_main", target_1), ("moka_pot_2_main", target_2)]
    else:
        order = [("moka_pot_2_main", target_1), ("moka_pot_1_main", target_2)]

    print(f"Order: {order[0][0]} -> {order[1][0]}")

    step_counter = 0
    for pot_name, target in order:
        obs, step_counter, done = pick_and_place(
            env, obs, traj, pot_name, target, step_counter=step_counter
        )
        if done:
            traj["success"] = True
            traj["total_steps"] = step_counter
            print(f"\n*** DONE (env reported success) steps={step_counter} ***")
            return traj

    success = env.check_success()
    traj["success"] = success
    traj["total_steps"] = step_counter

    if success:
        print(f"\n*** DEMO {demo_idx} SUCCEEDED! steps={step_counter} ***")
    else:
        print(f"\n--- Demo {demo_idx} FAILED steps={step_counter} ---")
        print(f"  Pot1: {np.round(_get_body_pos(env, 'moka_pot_1_main'), 4)}")
        print(f"  Pot2: {np.round(_get_body_pos(env, 'moka_pot_2_main'), 4)}")
        print(f"  Stove: {np.round(_get_body_pos(env, 'flat_stove_1_main'), 4)}")

    demo_dir = os.path.join(output_dir, f"demo_{demo_idx}")
    os.makedirs(demo_dir, exist_ok=True)
    render_obs(obs, step_idx=step_counter, output_dir=demo_dir)
    return traj


def main():
    num_demos = 5
    max_attempts = 25
    output_dir = "teleop_scratch/renders_collect"

    env, task_desc, init_states = create_env(use_swap=True)
    print(f"Task: {task_desc}")
    print(f"Init states: {len(init_states)}")

    successful = []
    init_idx = 0

    while len(successful) < num_demos and init_idx < max_attempts:
        traj = collect_one_demo(env, init_states, init_idx=init_idx,
                                demo_idx=len(successful), output_dir=output_dir)
        if traj["success"]:
            successful.append(traj)
            print(f"\n>>> {len(successful)}/{num_demos} demos collected")
        else:
            print(f"\n>>> Failed init_state={init_idx}, trying next...")
        init_idx += 1

    env.close()

    print(f"\n{'='*60}")
    if len(successful) >= num_demos:
        print(f"SUCCESS: Collected {num_demos} demos!")
    else:
        print(f"WARNING: Only collected {len(successful)}/{num_demos}")

    # Save
    for i, t in enumerate(successful):
        save_demo(t, demo_idx=i, task_description=task_desc)

    if successful:
        combined = save_demos_combined(successful, task_description=task_desc)
        verify_hdf5(combined)

    return successful


if __name__ == "__main__":
    main()
