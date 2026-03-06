"""Unpack a LIBERO-PRO HDF5 demo file into one HDF5 file per episode.

Each output file preserves the original structure exactly (all datasets,
attributes, and the top-level data-group attributes).

Usage:
    python scripts/unpack_hf5_episodes.py \
        --hdf5-path book_caddy_demo.hdf5 \
        --output-dir ./data_dumps/book_caddy_episodes
"""

import logging
import pathlib

import h5py
import tyro


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    """Recursively copy all datasets and attributes from src into dst."""
    # Copy group-level attributes
    for k, v in src.attrs.items():
        dst.attrs[k] = v

    for name, item in src.items():
        if isinstance(item, h5py.Group):
            child = dst.require_group(name)
            _copy_group(item, child)
        else:
            src.copy(name, dst)


def main(hdf5_path: str, output_dir: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    src_path = pathlib.Path(hdf5_path)
    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(src_path, "r") as src:
        data_group = src["data"]
        demo_keys = sorted(data_group.keys())
        top_attrs = dict(data_group.attrs)

        logging.info(f"Found {len(demo_keys)} episodes in {src_path.name}")

        for demo_key in demo_keys:
            out_path = out_dir / f"{src_path.stem}_{demo_key}.hdf5"
            with h5py.File(out_path, "w") as dst:
                # Recreate the same top-level data/ structure
                dst_data = dst.create_group("data")

                # Copy data-group level attributes (problem_info, etc.)
                for k, v in top_attrs.items():
                    dst_data.attrs[k] = v

                # Copy this demo's group with all its datasets and attributes
                dst_demo = dst_data.create_group(demo_key)
                _copy_group(data_group[demo_key], dst_demo)

            logging.info(f"  Wrote {demo_key} -> {out_path}")

    logging.info(f"Done. {len(demo_keys)} files written to {out_dir}")


if __name__ == "__main__":
    tyro.cli(main)
