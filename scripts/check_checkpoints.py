"""Check latest available checkpoints for all slurm scripts in a directory.

Usage:
    python scripts/check_checkpoints.py /home/skowshik/vla/codebase/openpi/slurm/libero10/1_demo
    python scripts/check_checkpoints.py /home/skowshik/vla/codebase/openpi/slurm/libero10/3_demos
    python scripts/check_checkpoints.py /home/skowshik/vla/codebase/openpi/slurm/libero10/5_demos
    python scripts/check_checkpoints.py slurm/robomimic
"""

import argparse
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from openpi.training.config import get_config


def extract_config_and_expname(script_path: pathlib.Path) -> tuple[str | None, str | None]:
    """Extract CONFIG_NAME and exp-name from a slurm/shell script."""
    text = script_path.read_text()

    # Match CONFIG_NAME="..." or CONFIG_NAME='...'
    m = re.search(r'CONFIG_NAME\s*=\s*["\']([^"\']+)["\']', text)
    config_name = m.group(1) if m else None

    # Match --exp-name=... (could use $CONFIG_NAME or a literal)
    m = re.search(r'--exp-name\s*=\s*(\S+)', text)
    if m:
        raw = m.group(1).strip('"').strip("'")
        # Resolve ${CONFIG_NAME} or $CONFIG_NAME references
        if config_name:
            raw = raw.replace("${CONFIG_NAME}", config_name).replace("$CONFIG_NAME", config_name)
        exp_name = raw
    else:
        exp_name = None

    return config_name, exp_name


def get_latest_checkpoint(ckpt_dir: pathlib.Path) -> int | None:
    """Find the largest numeric checkpoint step in a directory."""
    if not ckpt_dir.exists():
        return None
    steps = []
    for p in ckpt_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            steps.append(int(p.name))
    return max(steps) if steps else None


def main():
    parser = argparse.ArgumentParser(description="Check latest checkpoints for slurm scripts")
    parser.add_argument("slurm_dir", type=str, help="Directory containing slurm/shell scripts")
    args = parser.parse_args()

    slurm_dir = pathlib.Path(args.slurm_dir).resolve()
    if not slurm_dir.is_dir():
        print(f"Error: {slurm_dir} is not a directory")
        sys.exit(1)

    scripts = sorted(p for p in slurm_dir.iterdir() if p.suffix in (".slurm", ".sh"))
    if not scripts:
        print(f"No .slurm or .sh files found in {slurm_dir}")
        sys.exit(1)

    rows = []
    for script in scripts:
        config_name, exp_name = extract_config_and_expname(script)
        if not config_name:
            rows.append((script.name, "?", "no CONFIG_NAME found"))
            continue

        try:
            config = get_config(config_name)
        except ValueError:
            rows.append((script.name, config_name, "config not found"))
            continue

        if exp_name:
            ckpt_dir = pathlib.Path(config.checkpoint_base_dir).resolve() / config_name / exp_name
        else:
            ckpt_dir = pathlib.Path(config.checkpoint_base_dir).resolve() / config_name
            # Try to find any exp subdirectory
            if ckpt_dir.exists():
                subdirs = [d for d in ckpt_dir.iterdir() if d.is_dir()]
                if len(subdirs) == 1:
                    ckpt_dir = subdirs[0]

        latest = get_latest_checkpoint(ckpt_dir)
        if latest is not None:
            rows.append((script.name, config_name, f"{latest:,}"))
        else:
            rows.append((script.name, config_name, "no checkpoints"))

    # Print table
    col1 = max(len(r[0]) for r in rows)
    col2 = max(len(r[1]) for r in rows)
    col3 = max(len(r[2]) for r in rows)

    header = f"{'Script':<{col1}}  {'Config':<{col2}}  {'Latest Ckpt':<{col3}}"
    print(header)
    print("-" * len(header))
    for name, config, ckpt in rows:
        print(f"{name:<{col1}}  {config:<{col2}}  {ckpt:<{col3}}")


if __name__ == "__main__":
    main()
