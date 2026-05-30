# RoboCasa Tasks Reference — Complete List by Difficulty Tier

This document lists all RoboCasa tasks grouped by difficulty tier, and documents how to perturb robot initial pose and target object pose during evaluations.

---

## Table of Contents

1. [Task Overview](#1-task-overview)
2. [Atomic Tasks by Difficulty Tier](#2-atomic-tasks-by-difficulty-tier)
3. [Composite Tasks](#3-composite-tasks)
4. [Target Split Tasks (Evaluation)](#4-target-split-tasks-evaluation)
5. [Pose Perturbation During Evaluation](#5-pose-perturbation-during-evaluation)
6. [Eval Initialization Modes](#6-eval-initialization-modes)
7. [Configuration Reference](#7-configuration-reference)
8. [Example Evaluation Configurations](#8-example-evaluation-configurations)

---

## 1. Task Overview

| Category | Count | Description |
|----------|-------|-------------|
| Atomic tasks | 65 | Single-step manipulation (pick, place, open, close, turn, slide, etc.) |
| Composite tasks | 252 | Multi-step sequences (e.g., "prepare coffee" = multiple atomic steps) |
| Atomic target split | 18 | Atomic tasks with held-out object instances for evaluation |
| Composite seen target split | 16 | Composite tasks seen during training, with held-out objects |
| Composite unseen target split | 16 | Composite tasks NOT seen during training |

---

## 2. Atomic Tasks by Difficulty Tier

### Pick-and-Place Tasks (18 tasks)

#### Tier 1 — Easiest (Flat surface to flat surface)

| Task Name | Horizon | Target Split | Description |
|-----------|---------|:------------:|-------------|
| `PickPlaceCounterToCabinet` | 500 | Yes | Counter to open cabinet. Cabinet pre-opened. **Best starting point** — most tested, most data. |
| `PickPlaceCounterToStove` | 500 | Yes | Counter to stove. Wide, flat, clearly visible surface. |
| `PickPlaceSinkToCounter` | 500 | Yes | Sink to counter. Object starts in sink (lower surface), drops to counter. |
| `PickPlaceStoveToCounter` | 500 | No | Stove to counter. Object starts elevated on stove, return path. |

#### Tier 2 — Moderate (Enclosed target or occluded source)

| Task Name | Horizon | Target Split | Description |
|-----------|---------|:------------:|-------------|
| `PickPlaceDrawerToCounter` | 500 | Yes | Open drawer to counter. Source inside an enclosure. |
| `PickPlaceCabinetToCounter` | 300 | No | Open cabinet to counter. Source inside an enclosure. |
| `PickPlaceCounterToDrawer` | 500 | No | Counter to open drawer slot. Requires precision. |
| `PickPlaceCounterToMicrowave` | 700 | No | Counter to microwave. Pre-opened door, narrower opening. |
| `PickPlaceMicrowaveToCounter` | 500 | No | Microwave to counter. Extracting from narrow space. |

#### Tier 3 — Harder (Precision placement, unusual geometry)

| Task Name | Horizon | Target Split | Description |
|-----------|---------|:------------:|-------------|
| `PickPlaceCounterToSink` | 500 | No | Counter to sink. Sink is below counter level; reaching down. |
| `PickPlaceCounterToBlender` | 500 | No | Counter to blender jug. Small, oddly-shaped target. |
| `PickPlaceCounterToStandMixer` | 500 | No | Counter to stand mixer bowl. Small target. |
| `PickPlaceFridgeDrawerToShelf` | 500 | No | Fridge bottom drawer to fridge shelf. Inside fridge, two distinct regions. |
| `PickPlaceFridgeShelfToDrawer` | 500 | No | Fridge shelf to fridge bottom drawer. Inside fridge, two distinct regions. |

#### Tier 4 — Most Complex (Unusual target geometry, narrow access)

| Task Name | Horizon | Target Split | Description |
|-----------|---------|:------------:|-------------|
| `PickPlaceCounterToOven` | 500 | No | Counter to oven rack. Reaching into oven interior. |
| `PickPlaceCounterToToasterOven` | 500 | No | Counter to toaster oven interior. Small door, narrow access. |
| `PickPlaceToasterOvenToCounter` | 500 | No | Toaster oven to counter. Extracting from toaster oven. |
| `PickPlaceToasterToCounter` | 500 | Yes | Toaster slot to counter. Narrow slot is a precision challenge. |

---

### Open/Close Tasks (~22 tasks)

#### Tier 1 — Easiest (Single hinge/slider motion)

| Task Name | Description |
|-----------|-------------|
| `OpenCabinet` | Open a cabinet door (single hinge joint) |
| `CloseCabinet` | Close a cabinet door |
| `OpenDrawer` | Pull open a drawer (linear slider) |
| `CloseDrawer` | Push closed a drawer |
| `OpenFridge` | Open refrigerator door |
| `CloseFridge` | Close refrigerator door |

#### Tier 2 — Moderate (Heavier doors, less common fixtures)

| Task Name | Description |
|-----------|-------------|
| `OpenOven` | Open oven door (heavy, swings down) |
| `CloseOven` | Close oven door |
| `OpenMicrowave` | Open microwave door |
| `CloseMicrowave` | Close microwave door |
| `OpenDishwasher` | Open dishwasher door (front-loading) |
| `CloseDishwasher` | Close dishwasher door |
| `OpenFridgeDrawer` | Open fridge bottom drawer |
| `CloseFridgeDrawer` | Close fridge bottom drawer |

#### Tier 3 — Harder (Small handles, precision required)

| Task Name | Description |
|-----------|-------------|
| `OpenBlenderLid` | Open blender lid (small target) |
| `CloseBlenderLid` | Close blender lid |
| `OpenElectricKettleLid` | Open electric kettle lid |
| `CloseElectricKettleLid` | Close electric kettle lid |
| `OpenStandMixerHead` | Tilt stand mixer head back |
| `CloseStandMixerHead` | Tilt stand mixer head forward |
| `OpenToasterOvenDoor` | Open toaster oven door (small, precise) |
| `CloseToasterOvenDoor` | Close toaster oven door |

---

### Turn/Toggle Tasks (~11 tasks)

#### Tier 1 — Easiest (Large knobs/buttons)

| Task Name | Description |
|-----------|-------------|
| `TurnOnStove` | Turn stove burner knob to on |
| `TurnOffStove` | Turn stove burner knob to off |
| `TurnOnMicrowave` | Press microwave start button |
| `TurnOffMicrowave` | Press microwave stop button |

#### Tier 2 — Moderate (Faucet/appliance controls)

| Task Name | Description |
|-----------|-------------|
| `TurnOnSinkFaucet` | Turn sink faucet handle to on |
| `TurnOffSinkFaucet` | Turn sink faucet handle to off |
| `TurnSinkSpout` | Rotate sink spout direction |
| `AdjustWaterTemperature` | Adjust water temperature dial |

#### Tier 3 — Harder (Small appliances, precision)

| Task Name | Description |
|-----------|-------------|
| `TurnOnToaster` | Press toaster lever down |
| `TurnOnToasterOven` | Press toaster oven start |
| `TurnOnElectricKettle` | Toggle electric kettle on |
| `TurnOnBlender` | Turn on blender |
| `AdjustToasterOvenTemperature` | Adjust toaster oven temperature dial |
| `LowerHeat` | Lower stove heat (fine adjustment) |

---

### Slide Tasks (3 tasks)

| Task Name | Tier | Description |
|-----------|------|-------------|
| `SlideDishwasherRack` | Moderate | Pull/push dishwasher rack |
| `SlideOvenRack` | Moderate | Pull/push oven rack |
| `SlideToasterOvenRack` | Harder | Pull/push small toaster oven rack |

---

### Coffee/Beverage Tasks (2 tasks)

| Task Name | Tier | Description |
|-----------|------|-------------|
| `CoffeeSetupMug` | Moderate | Place mug under coffee machine |
| `CoffeeServeMug` | Moderate | Serve prepared coffee mug |

---

### Specialized Tasks

| Task Name | Tier | Description |
|-----------|------|-------------|
| `NavigateKitchen` | Easiest | Navigate robot to a target location (base movement only) |
| `PreheatOven` | Moderate | Turn oven to correct temperature |
| `StartCoffeeMachine` | Moderate | Start a coffee machine |
| `CheesyBread` | Harder | Prepare cheesy bread |
| `MakeIcedCoffee` | Harder | Multi-step iced coffee preparation |
| `PackDessert` | Harder | Pack dessert items into container |

---

## 3. Composite Tasks

Composite tasks chain multiple atomic operations. They are significantly harder due to long horizons and multi-step reasoning.

### Composite Seen Tasks (16 — seen during training, target split available)

| Task Name | Description |
|-----------|-------------|
| `PrepareCoffee` | Full coffee preparation sequence |
| `DeliverStraw` | Get straw and deliver to drink |
| `GetToastedBread` | Toast bread and retrieve |
| `KettleBoiling` | Boil water using kettle |b
| `LoadDishwasher` | Load dishes into dishwasher |
| `PackIdenticalLunches` | Pack matching lunch items |
| `PreSoakPan` | Fill pan with water for soaking |
| `RinseSinkBasin` | Clean out the sink basin |
| `ScrubCuttingBoard` | Scrub and clean cutting board |
| `SearingMeat` | Sear meat on stove |
| `SetUpCuttingStation` | Prepare cutting board and tools |
| `StackBowlsCabinet` | Stack bowls and store in cabinet |
| `SteamInMicrowave` | Steam food using microwave |
| `StirVegetables` | Stir vegetables in pan |
| `StoreLeftoversInBowl` | Transfer leftovers to storage bowl |
| `WashLettuce` | Wash lettuce in sink |

### Composite Unseen Tasks (16 — NOT seen during training)

| Task Name | Description |
|-----------|-------------|
| `ArrangeBreadBasket` | Arrange bread items in basket |
| `ArrangeTea` | Set up tea service |
| `BreadSelection` | Select and arrange bread |
| `CategorizeCondiments` | Sort condiments by type |
| `CuttingToolSelection` | Select appropriate cutting tools |
| `GarnishPancake` | Add garnish to pancake |
| `GatherTableware` | Collect tableware from various locations |
| `HeatKebabSandwich` | Heat kebab sandwich |
| `MakeIceLemonade` | Prepare iced lemonade |
| `PanTransfer` | Transfer food between pans |
| `PortionHotDogs` | Portion hot dogs onto plates |
| `RecycleBottlesByType` | Sort bottles for recycling |
| `SeparateFreezerRack` | Organize freezer rack items |
| `WaffleReheat` | Reheat waffles |
| `WashFruitColander` | Wash fruit in colander |
| `WeighIngredients` | Weigh ingredients on scale |

---

## 4. Target Split Tasks (Evaluation)

The **target split** uses held-out object instances (different meshes/textures from training) and clutter for evaluating generalization. Each task has ~500 human demos across 10 (layout, style) pairs.

### 18 Atomic Target Split Tasks

| Task Name | Category |
|-----------|----------|
| `CloseBlenderLid` | Close |
| `CloseFridge` | Close |
| `CloseToasterOvenDoor` | Close |
| `CoffeeSetupMug` | Coffee |
| `NavigateKitchen` | Navigation |
| `OpenCabinet` | Open |
| `OpenDrawer` | Open |
| `OpenStandMixerHead` | Open |
| `PickPlaceCounterToCabinet` | Pick-and-Place |
| `PickPlaceCounterToStove` | Pick-and-Place |
| `PickPlaceDrawerToCounter` | Pick-and-Place |
| `PickPlaceSinkToCounter` | Pick-and-Place |
| `PickPlaceToasterToCounter` | Pick-and-Place |
| `SlideDishwasherRack` | Slide |
| `TurnOffStove` | Turn |
| `TurnOnElectricKettle` | Turn |
| `TurnOnMicrowave` | Turn |
| `TurnOnSinkFaucet` | Turn |

---

## 5. Pose Perturbation During Evaluation

Perturbation adds controlled noise to the robot's initial pose and/or the target object's pose at the start of each evaluation episode, introducing variation to test policy robustness.

### 5.1 Robot Pose Perturbation

**What is perturbed:** The robot's mobile base position (X, Y) and yaw orientation.

**Parameter:** `eval_robot_pose_noise` (float) — applies uniform noise in range `[-N, +N]` where N is in **meters** for XY and **radians** for yaw.

**Two application paths depending on eval mode:**

#### Path A: Via ep_meta fields (soft reset)

Used in modes: `fixture_pair_fresh_placement`, `fixture_pair_same_category`, `fixture_pair_object_pool`.

The controller modifies `init_robot_base_pos` and `init_robot_base_ori` in the episode metadata before environment reset:

```python
# In RoboCasaEvalResetController._apply_robot_pose_policy()
if keep_robot_pose and robot_pose_noise > 0.0:
    pos = np.array(ep_meta["init_robot_base_pos"], dtype=float)
    ori = np.array(ep_meta["init_robot_base_ori"], dtype=float)
    pos[:2] += rng.uniform(-robot_pose_noise, robot_pose_noise, size=2)  # XY noise (meters)
    ori[2]  += rng.uniform(-robot_pose_noise, robot_pose_noise)          # Yaw noise (radians)
    ep_meta["init_robot_base_pos"] = pos.tolist()
    ep_meta["init_robot_base_ori"] = ori.tolist()
```

If `keep_robot_pose=False`, the fields are removed entirely and the Kitchen environment resamples a fresh random robot pose at reset.

#### Path B: Direct MuJoCo qpos modification (hard reset)

Used in mode: `exact_state_replay` after restoring the full simulator state.

```python
# In gym_wrapper.py apply_robot_pose_noise()
# Identifies mobile base joints by name:
#   mobilebase0_joint_mobile_forward (X)
#   mobilebase0_joint_mobile_side (Y)
#   mobilebase0_joint_mobile_yaw (rotation)
for jname in base_joints:
    addr = sim.model.get_joint_qpos_addr(jname)
    sim.data.qpos[addr] += rng.uniform(-noise, noise)
sim.forward()  # Propagate physics
```

### 5.2 Object Position Perturbation

**What is perturbed:** The target object's XY position on its placement surface.

**Parameter:** `eval_object_pose_noise` (float) — uniform noise in **meters** on X and Y.

**Applied only in `exact_state_replay` mode** via direct MuJoCo qpos modification:

```python
# Free joint qpos layout: [x, y, z, qw, qx, qy, qz]
if object_pose_noise > 0.0:
    sim.data.qpos[qpos_addr:qpos_addr+2] += rng.uniform(
        -object_pose_noise, object_pose_noise, size=2
    )
    # Z (height) is NOT perturbed — object stays on surface
```

### 5.3 Object Orientation Perturbation

**What is perturbed:** The target object's yaw (rotation around Z-axis).

**Parameter:** `eval_object_ori_noise` (float) — uniform noise in **radians** on yaw.

**Applied only in `exact_state_replay` mode** via quaternion composition:

```python
if object_ori_noise > 0.0:
    yaw = rng.uniform(-object_ori_noise, object_ori_noise)
    # Build Z-axis rotation quaternion
    dq = [cos(yaw/2), 0, 0, sin(yaw/2)]
    # Compose with existing object quaternion: q_new = dq * q_current
    new_q = quaternion_multiply(dq, current_q)
    sim.data.qpos[qpos_addr+3:qpos_addr+7] = new_q
```

### 5.4 Perturbation Summary

| Component | Parameter | Unit | Affects | Eval Modes |
|-----------|-----------|------|---------|------------|
| Robot XY position | `eval_robot_pose_noise` | meters | Base X, Y | All modes |
| Robot yaw | `eval_robot_pose_noise` | radians | Base heading | All modes |
| Object XY position | `eval_object_pose_noise` | meters | Object X, Y | `exact_state_replay` only |
| Object yaw | `eval_object_ori_noise` | radians | Object Z-rotation | `exact_state_replay` only |

**Note:** For non-exact-state-replay modes, object pose variation comes naturally from the environment's randomized object placement — no explicit noise parameter is needed.

---

## 6. Eval Initialization Modes

The `RoboCasaEvalResetController` (from `dsrl_pi0/examples/robocasa_eval_reset.py`) controls how the environment is initialized at the start of each evaluation episode.

### Mode 1: `exact_state_replay`

- Restores the **exact MuJoCo simulator state** from a recorded demo (loads `states.npz`, `model.xml.gz`, `ep_meta.json`)
- Round-robin cycles through pool episodes
- Robot and object pose noise applied **after** state restoration
- **Use for:** Exact reproducibility with controlled perturbations

### Mode 2: `fixture_pair_fresh_placement`

- Keeps the same scene fixtures (counter name, cabinet name) from the recorded demo
- Keeps the same object config (same mesh/instance)
- RoboCasa resamples a fresh random placement position within the fixture's placement region
- Robot pose: controlled by `keep_robot_pose` + `robot_pose_noise`
- **Use for:** Object placement variation with identical objects

### Mode 3: `fixture_pair_same_category`

- Keeps the same scene fixtures and object **category** (e.g., "fruit")
- Removes `mjcf_path` from object config so RoboCasa samples a **different instance** of the same category (different mesh/texture)
- Resamples object placement position
- Robot pose: controlled by noise parameters
- **Use for:** Object instance variation within the same category

### Mode 4: `fixture_pair_object_pool`

- Keeps the same scene fixtures only
- Removes `object_cfgs` entirely — RoboCasa samples **completely random objects**
- Resamples placement
- Robot pose: controlled by noise parameters
- **Use for:** Full object randomization (generalization testing)

### Comparison Table

| Feature | exact_state_replay | fixture_pair_fresh_placement | fixture_pair_same_category | fixture_pair_object_pool |
|---------|:--:|:--:|:--:|:--:|
| Scene fixtures | Exact | Same | Same | Same |
| Object instance | Exact | Same | Different (same category) | Random |
| Object placement | Exact (+noise) | Random | Random | Random |
| Robot pose | Exact (+noise) | Recorded or random (+noise) | Recorded or random (+noise) | Recorded or random (+noise) |
| Explicit object noise params | Yes | N/A (already random) | N/A (already random) | N/A (already random) |

---

## 7. Configuration Reference

### Training Config Fields (`LeRobotRobocasaDataConfig`)

Located in `src/openpi/training/config.py`:

```python
@dataclasses.dataclass(frozen=True)
class LeRobotRobocasaDataConfig(DataConfigFactory):
    # ... data fields ...

    # Eval initialization
    eval_init_mode: str | None = None
    # Options: "exact_state_replay", "fixture_pair_fresh_placement",
    #          "fixture_pair_same_category", "fixture_pair_object_pool"

    # Eval pool configuration
    eval_pool_episode_ids: list[int] | None = None
    eval_pool_fixture_refs: dict[str, str] | None = None
    eval_pool_object_categories: list[str] | None = None

    # Robot pose perturbation
    eval_keep_robot_pose: bool = False       # Keep recorded robot pose (True) or resample (False)
    eval_robot_pose_noise: float = 0.0       # Uniform noise on robot XY (meters) + yaw (radians)

    # Object pose perturbation (exact_state_replay only)
    eval_object_pose_noise: float = 0.0      # Uniform noise on object XY (meters)
    eval_object_ori_noise: float = 0.0       # Uniform noise on object yaw (radians)
```

### Eval Script Arguments (`examples/robocasa/main.py`)

```
--eval-init-mode          One of the 4 modes above
--dataset-path            Path to the LeRobot dataset for loading eval episodes
--eval-pool-episode-ids   JSON list of episode IDs to cycle through
--eval-pool-fixture-refs  JSON dict of fixture name→value pairs
--eval-pool-object-categories  JSON list of object categories
--eval-keep-robot-pose    Whether to keep recorded robot pose
--eval-robot-pose-noise   Float, meters/radians
--eval-object-pose-noise  Float, meters
--eval-object-ori-noise   Float, radians
```

---

## 8. Example Evaluation Configurations

### Exact Replay, No Perturbation (Baseline)

Reproduce training demos exactly — useful for verifying the model can replicate seen states.

```python
eval_init_mode = "exact_state_replay"
eval_keep_robot_pose = True
eval_robot_pose_noise = 0.0
eval_object_pose_noise = 0.0
eval_object_ori_noise = 0.0
```

### Exact Replay + Small Perturbation

Test robustness to small deviations from training distribution.

```python
eval_init_mode = "exact_state_replay"
eval_keep_robot_pose = True
eval_robot_pose_noise = 0.025    # 2.5 cm robot XY + 0.025 rad (~1.4 deg) yaw
eval_object_pose_noise = 0.025   # 2.5 cm object XY
eval_object_ori_noise = 0.5      # ~28.6 degrees object yaw
```

### Exact Replay + Large Perturbation (Stress Test)

```python
eval_init_mode = "exact_state_replay"
eval_keep_robot_pose = True
eval_robot_pose_noise = 0.1      # 10 cm robot XY + 0.1 rad (~5.7 deg) yaw
eval_object_pose_noise = 0.1     # 10 cm object XY
eval_object_ori_noise = 0.5      # ~28.6 degrees object yaw
```

### Same Category, New Instance (Generalization)

Same fixtures and object category, but different object mesh and random placement.

```python
eval_init_mode = "fixture_pair_same_category"
eval_keep_robot_pose = True
eval_robot_pose_noise = 0.025    # Small robot pose variation
# Object variation comes from resampled placement — no explicit noise needed
```

### Full Randomization (Maximum Generalization)

Random objects, random placement, random robot pose.

```python
eval_init_mode = "fixture_pair_object_pool"
eval_keep_robot_pose = False     # Robot pose fully resampled
# Everything else is random — no noise parameters needed
```

### Recommended Perturbation Sweep

To systematically evaluate robustness, sweep over noise levels:

| Level | `robot_pose_noise` | `object_pose_noise` | `object_ori_noise` |
|-------|:------------------:|:-------------------:|:------------------:|
| None | 0.0 | 0.0 | 0.0 |
| Low | 0.025 (2.5 cm) | 0.025 (2.5 cm) | 0.25 (~14 deg) |
| Medium | 0.05 (5 cm) | 0.05 (5 cm) | 0.5 (~29 deg) |
| High | 0.1 (10 cm) | 0.1 (10 cm) | 1.0 (~57 deg) |

---

## 9. Key Code Files

| File | Purpose |
|------|---------|
| `src/openpi/training/config.py` | Training config with eval perturbation parameters (`LeRobotRobocasaDataConfig`) |
| `examples/robocasa/main.py` | Evaluation script — connects to model server, runs rollouts with perturbation |
| `dsrl_pi0/examples/robocasa_eval_reset.py` | `RoboCasaEvalResetController` — implements all 4 eval init modes + robot pose policy |
| `openpi_robocasa/robocasa/robocasa/wrappers/gym_wrapper.py` | Gym wrapper — applies robot/object pose noise directly in MuJoCo |
| `openpi_robocasa/robocasa/robocasa/utils/dataset_registry.py` | Task registry — `ATOMIC_TASK_DATASETS`, `COMPOSITE_TASK_DATASETS`, `TARGET_TASKS` |

---

## 10. Task Selection for Online RL with Pose Perturbations

**Goal:** Select multi-step tasks that contain at least one pick-and-place sub-step, with horizon < 800, suitable for online RL with position and orientation perturbations on the initial state.

### 10.1 Key Constraint: No Composite Task Has Horizon Strictly < 800

All composite tasks have horizon >= 800. The full composite task horizon distribution:

| Horizon | Tasks |
|---------|-------|
| 800 | CuttingToolSelection, ScrubCuttingBoard |
| 900 | RinseSinkBasin |
| 1000 | KettleBoiling |
| 1100 | CategorizeCondiments, WashLettuce |
| 1200 | PrepareCoffee, LoadDishwasher, PanTransfer |
| 1300 | BreadSelection |
| 1400 | StackBowlsCabinet, SteamInMicrowave |
| 1500 | ArrangeTea, GatherTableware, PortionHotDogs |
| 1600 | PreSoakPan, SetUpCuttingStation, StirVegetables, SeparateFreezerRack |
| 1700 | DeliverStraw, StoreLeftoversInBowl |
| 1800 | GarnishPancake, HeatKebabSandwich |
| 1900 | RecycleBottlesByType |
| 2000 | GetToastedBread, MakeIceLemonade, WeighIngredients |
| 2100 | WashFruitColander |
| 2600 | PackIdenticalLunches |
| 2700 | WaffleReheat |
| 2900 | SearingMeat, ArrangeBreadBasket |

### 10.2 Recommended Tasks (Horizon <= 800, Multi-Step, With Pick-and-Place)

These two composite tasks sit exactly at the 800 boundary and satisfy all criteria:

#### 1. `CuttingToolSelection` — Horizon: 800

- **Sub-steps:** Pick the appropriate cutting tool (knife) → Place it on the cutting board next to the fruit/vegetable.
- **Pick-and-place:** Yes — picking a tool and placing it on a target surface.
- **Why good for RL with perturbations:**
  - Object (cutting tool) position perturbation directly changes the reach/grasp strategy.
  - Object orientation perturbation matters because knives have elongated geometry — grasp point depends on yaw.
  - Placing on the cutting board requires precision, so robot base perturbation tests recovery.
- **Perturbation targets:** Cutting tool position/orientation on counter, robot base pose.

#### 2. `ScrubCuttingBoard` — Horizon: 800

- **Sub-steps:** Pick up the sponge from the counter → Bring it to the cutting board → Scrub the cutting board.
- **Pick-and-place:** Yes — the sponge must be picked and brought to the board (pick-and-manipulate).
- **Why good for RL with perturbations:**
  - Sponge position perturbation changes the initial reach.
  - Sponge orientation perturbation is less impactful (symmetric shape), but cutting board position can also vary.
  - Scrubbing requires sustained contact after placement, adding a manipulation phase beyond pure pick-and-place.
- **Perturbation targets:** Sponge position on counter, robot base pose.

### 10.3 Near-Boundary Candidates (Horizon 900–1200)

If the horizon constraint can be relaxed slightly, these tasks offer richer multi-step structure with clear pick-and-place:

| Task | Horizon | Sub-Steps | Pick-and-Place Description | Perturbation Suitability |
|------|---------|-----------|---------------------------|--------------------------|
| `KettleBoiling` | 1000 | 3 | Pick kettle from counter → Place on stove burner → Turn burner on | Kettle position/orientation perturbation changes grasp; placement on burner requires precision. Strong candidate. |
| `CategorizeCondiments` | 1100 | 2+ | Pick shaker and condiment bottle from counter → Place next to counterparts in cabinet | Multiple pick-and-place steps; object perturbation directly affects each grasp. |
| `PrepareCoffee` | 1200 | 3 | Pick mug from cabinet → Place under coffee machine dispenser → Press start button | Mug position perturbation in cabinet changes extraction strategy; target split available. |
| `LoadDishwasher` | 1200 | 3 | Pick dishes from counter → Place in dishwasher → Close dishwasher door | Multiple objects to pick; dishwasher placement requires precision. |
| `PanTransfer` | 1200 | 3 | Pick pan → Dump vegetables onto plate → Return pan to stove | Pick-and-place + pouring; pan orientation perturbation affects grip strategy. |

### 10.4 Recommended Configuration for Online RL

For the selected tasks, use `exact_state_replay` mode with perturbations to create a distribution of initial states around demonstrated trajectories:

```python
# Recommended starting perturbation levels for online RL
eval_init_mode = "exact_state_replay"
eval_keep_robot_pose = True

# Start with moderate perturbations, increase as policy improves
eval_robot_pose_noise = 0.05     # 5 cm robot XY + ~2.9 deg yaw
eval_object_pose_noise = 0.05    # 5 cm object XY
eval_object_ori_noise = 0.5      # ~28.6 deg object yaw
```

### 10.5 Summary

| Priority | Task | Horizon | # Sub-Steps | Pick-and-Place | Best For |
|----------|------|---------|-------------|----------------|----------|
| 1 | `CuttingToolSelection` | 800 | 2 | Pick tool → Place on board | Orientation-sensitive grasping |
| 2 | `ScrubCuttingBoard` | 800 | 3 | Pick sponge → Scrub board | Pick + sustained manipulation |
| 3 | `KettleBoiling` | 1000 | 3 | Pick kettle → Place on stove → Turn on | Multi-step with diverse sub-tasks |
| 4 | `CategorizeCondiments` | 1100 | 2+ | Pick items → Place in cabinet | Multiple pick-and-place instances |
| 5 | `PrepareCoffee` | 1200 | 3 | Pick mug → Place under machine → Start | Target split available for generalization |



/home/skowshik/vla/codebase/openpi/robocasa_vis/KettleBoiling.mp4
/home/skowshik/vla/codebase/openpi/robocasa_vis/WashLettuce.mp4


