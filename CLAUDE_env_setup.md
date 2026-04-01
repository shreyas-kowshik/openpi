# OpenPI Training Environment Setup (Clean Reproduction)

This document describes how to set up a clean environment for running openpi training
(libero10, robocasa, etc.) from scratch.

---

## Prerequisites

- Linux (tested on RHEL 9 / Rocky 9, kernel 5.14+)
- CUDA 12.x compatible GPU(s)
- `git`, `conda` (or `mamba`)
- `uv` (the Python package manager) — install with `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Verified Versions

| Component        | Version    |
|------------------|------------|
| Python           | 3.11.x     |
| uv               | 0.9.7+     |
| JAX              | 0.5.3      |
| PyTorch          | 2.7.1+cu126|
| Flax             | 0.10.2     |
| NumPy            | 1.26.4     |
| MuJoCo           | 3.3.1      |
| robosuite        | 1.5.2      |
| robocasa         | 1.0.0      |

---

## Step 1: Clone Repositories

```bash
# Pick a workspace root
WORKSPACE="$HOME/vla/codebase"
mkdir -p "$WORKSPACE"

# Clone openpi
cd "$WORKSPACE"
git clone https://github.com/Physical-Intelligence/openpi.git
cd openpi

# Clone robocasa (needed for groot_utils dataset loading, even for libero10)
cd "$WORKSPACE"
git clone https://github.com/robocasa/robocasa.git
# Note: robocasa is at $WORKSPACE/robocasa
```

## Step 2: Set Up the uv Virtual Environment

openpi uses `uv` to manage its virtualenv (`.venv/`) and dependencies. All `uv run` commands
use this `.venv`, **not** your system Python or conda env.

```bash
cd "$WORKSPACE/openpi"

# Create the venv and install all openpi dependencies (from pyproject.toml + uv.lock)
uv sync
```

This creates `.venv/` with Python 3.11 and installs all dependencies (JAX, PyTorch, Flax,
lerobot, etc.) as specified in `pyproject.toml` and `uv.lock`.

## Step 3: Install Extra Dependencies (robocasa + robosuite + mujoco)

openpi's `config.py` unconditionally imports `robocasa.utils.groot_utils`, which in turn
requires `robosuite` and `mujoco==3.3.1`. These are **not** in openpi's `pyproject.toml`
by default, so you must install them manually.

### 3a. Add mujoco override to pyproject.toml

The uv lockfile pins mujoco to 2.3.7 (via gym-aloha). To force mujoco 3.3.1, add it to
the override-dependencies:

```bash
cd "$WORKSPACE/openpi"
```

Edit `pyproject.toml` and find this line:
```toml
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74"]
```

Change it to:
```toml
override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "mujoco==3.3.1"]
```

Then re-lock:
```bash
uv lock
```

### 3b. Install robosuite and robocasa into the uv venv

```bash
cd "$WORKSPACE/openpi"

# Install robosuite (no-deps to avoid pulling conflicting numpy/mujoco versions)
uv pip install robosuite --no-deps

# Install robocasa from local clone (no-deps for same reason)
uv pip install --no-deps -e "$WORKSPACE/robocasa"
```

> **Why `--no-deps`?** robocasa declares `numpy==2.2.5` and `lerobot==0.3.3` as hard
> requirements, but openpi needs `numpy<2.0.0` and a specific lerobot git commit. The
> core functionality (groot_utils dataset loading) works fine with openpi's numpy 1.26.x.

### 3c. Verify installations

```bash
cd "$WORKSPACE/openpi"

uv run python -c "
import robocasa; print('robocasa:', robocasa.__version__)
import robosuite; print('robosuite:', robosuite.__version__)
import mujoco; print('mujoco:', mujoco.__version__)
import jax; print('jax:', jax.__version__)
import torch; print('torch:', torch.__version__)
import numpy; print('numpy:', numpy.__version__)
print('All imports OK')
"
```

Expected output (ignoring warnings):
```
robocasa: 1.0.0
robosuite: 1.5.2
mujoco: 3.3.1
jax: 0.5.3
torch: 2.7.1+cu126
numpy: 1.26.4
All imports OK
```

> **Note**: You will see harmless warnings about robosuite's missing private macro file,
> robosuite_models, and mink-based IK. These do not affect training.

## Step 4: Configure Environment Variables

Create a `.env` file in the openpi repo root:

```bash
cd "$WORKSPACE/openpi"
cat > .env << 'EOF'
HF_HOME="/data/user_data/$USER/huggingface"
EOF
```

The slurm scripts source this file automatically. Adjust `HF_HOME` to wherever you want
HuggingFace datasets/models cached.

Also set the HF datasets cache in slurm scripts (already done in the provided scripts):
```bash
export HF_DATASETS_CACHE="/data/hf_cache/datasets/"
```

## Step 5: Create Log Directory

```bash
mkdir -p "/data/user_data/$USER/openpi-libero10/logs"
```

## Step 6: Run Training

### Direct execution (no SLURM)

```bash
cd "$WORKSPACE/openpi"

# Compute norm stats first
uv run scripts/compute_norm_stats.py --config-name pi05_libero10_alphabet_soup_cream_cheese_basket_ep1

# Train
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
    pi05_libero10_alphabet_soup_cream_cheese_basket_ep1 \
    --exp-name=pi05_libero10_alphabet_soup_cream_cheese_basket_ep1-v1 \
    --resume
```

### Via SLURM

```bash
cd "$WORKSPACE/openpi"
bash slurm/libero10/1_demo/alphabet_soup_cream_cheese_basket.slurm
# Or: sbatch slurm/libero10/1_demo/alphabet_soup_cream_cheese_basket.slurm
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'robocasa'`
robocasa is not installed in the `.venv`. Run:
```bash
uv pip install --no-deps -e /path/to/robocasa
```

### `ModuleNotFoundError: No module named 'robosuite'`
robosuite is not installed in the `.venv`. Run:
```bash
uv pip install robosuite --no-deps
```

### `AssertionError: MuJoCo version must be 3.3.1`
The uv lockfile is overriding mujoco back to 2.3.7. Make sure you added `"mujoco==3.3.1"`
to `override-dependencies` in `pyproject.toml` and ran `uv lock` (see Step 3a).

### `uv run` reverts manually-installed packages
`uv run` syncs the environment from the lockfile before running. Any packages not in
`pyproject.toml` or installed via `uv pip install` will be removed. That's why:
- mujoco must be overridden via `override-dependencies` in `pyproject.toml`
- robosuite and robocasa are installed via `uv pip install` (which uv respects)

### numpy version warning from robocasa
```
UserWarning: robocasa recommends numpy==2.2.5 but found 1.26.4
```
This is harmless. The groot_utils data loading works correctly with numpy 1.26.x.

### robosuite private macro file warning
```
[robosuite WARNING] No private macro file found!
```
Harmless for training. To silence it, run:
```bash
uv run python $(uv run python -c "import robosuite; import os; print(os.path.join(os.path.dirname(robosuite.__file__), 'scripts', 'setup_macros.py'))")
```

### LeRobot dataset version warning
```
The dataset you requested (physical-intelligence/libero) is in 2.0 format.
```
This is a warning from lerobot about a format version mismatch. It is backward-compatible
and does not affect training.

---

## Quick Reference: One-Shot Setup Script

```bash
#!/bin/bash
# Clean environment setup for openpi training
# Usage: bash setup_openpi_env.sh /path/to/workspace /path/to/robocasa

set -euo pipefail

WORKSPACE="${1:?Usage: $0 <workspace_dir> <robocasa_dir>}"
ROBOCASA_DIR="${2:?Usage: $0 <workspace_dir> <robocasa_dir>}"

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# 2. Clone openpi if not present
if [ ! -d "$WORKSPACE/openpi" ]; then
    git clone https://github.com/Physical-Intelligence/openpi.git "$WORKSPACE/openpi"
fi
cd "$WORKSPACE/openpi"

# 3. Add mujoco override if not already present
if ! grep -q 'mujoco==3.3.1' pyproject.toml; then
    sed -i 's/override-dependencies = \["ml-dtypes==0.4.1", "tensorstore==0.1.74"\]/override-dependencies = ["ml-dtypes==0.4.1", "tensorstore==0.1.74", "mujoco==3.3.1"]/' pyproject.toml
fi

# 4. Sync uv environment
uv lock
uv sync

# 5. Install robosuite and robocasa
uv pip install robosuite --no-deps
uv pip install --no-deps -e "$ROBOCASA_DIR"

# 6. Create .env if not present
if [ ! -f .env ]; then
    echo 'HF_HOME="/data/user_data/'"$USER"'/huggingface"' > .env
fi

# 7. Create log directory
mkdir -p "/data/user_data/$USER/openpi-libero10/logs"

echo "Setup complete. Verify with:"
echo "  cd $WORKSPACE/openpi"
echo '  uv run python -c "import robocasa, robosuite, mujoco; print(\"OK\")"'
```
