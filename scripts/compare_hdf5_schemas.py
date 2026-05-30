"""Compare the schemas of two HDF5 files side-by-side.

Prints groups, datasets (shape, dtype), and attributes for each file,
then highlights any differences.

Usage:
    python scripts/compare_hdf5_schemas.py \
        --file-a /path/to/first.hdf5 \
        --file-b /path/to/second.hdf5
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import h5py


@dataclass
class NodeInfo:
    """Schema info for a single HDF5 node (group or dataset)."""
    kind: str  # "group" or "dataset"
    shape: Optional[tuple] = None
    dtype: Optional[str] = None
    attrs: dict = field(default_factory=dict)


def collect_schema(g: h5py.Group, prefix: str = "") -> dict[str, NodeInfo]:
    """Recursively collect schema info for every node under g."""
    schema: dict[str, NodeInfo] = {}
    for key in sorted(g.keys()):
        path = f"{prefix}/{key}" if prefix else key
        item = g[key]
        if isinstance(item, h5py.Group):
            schema[path] = NodeInfo(kind="group", attrs={k: _fmt_attr(v) for k, v in item.attrs.items()})
            schema.update(collect_schema(item, path))
        else:
            schema[path] = NodeInfo(
                kind="dataset",
                shape=item.shape,
                dtype=str(item.dtype),
                attrs={k: _fmt_attr(v) for k, v in item.attrs.items()},
            )
    return schema


def _fmt_attr(v) -> str:
    """Format an attribute value for display."""
    if isinstance(v, bytes):
        return v.decode(errors="replace")
    return repr(v)


def _node_str(info: NodeInfo) -> str:
    if info.kind == "group":
        return "group"
    return f"shape={info.shape}, dtype={info.dtype}"


def print_schema(label: str, path: str, schema: dict[str, NodeInfo], top_attrs: dict):
    print(f"\n{'='*70}")
    print(f"  {label}: {path}")
    print(f"{'='*70}")
    if top_attrs:
        print(f"  (root attrs) {top_attrs}")
    for key, info in sorted(schema.items()):
        indent = "  " * (key.count("/") + 1)
        print(f"{indent}{key}: {_node_str(info)}")
        if info.attrs:
            print(f"{indent}  attrs: {info.attrs}")


def compare(schema_a: dict[str, NodeInfo], schema_b: dict[str, NodeInfo],
            top_attrs_a: dict, top_attrs_b: dict,
            label_a: str, label_b: str):
    keys_a = set(schema_a.keys())
    keys_b = set(schema_b.keys())
    all_keys = sorted(keys_a | keys_b)

    diffs = []

    # Top-level attrs
    all_top_attr_keys = sorted(set(top_attrs_a.keys()) | set(top_attrs_b.keys()))
    for ak in all_top_attr_keys:
        va = top_attrs_a.get(ak)
        vb = top_attrs_b.get(ak)
        if va != vb:
            diffs.append(f"  root attr '{ak}': {label_a}={va}  vs  {label_b}={vb}")

    for key in all_keys:
        in_a = key in keys_a
        in_b = key in keys_b

        if in_a and not in_b:
            diffs.append(f"  {key}: only in {label_a} ({_node_str(schema_a[key])})")
            continue
        if in_b and not in_a:
            diffs.append(f"  {key}: only in {label_b} ({_node_str(schema_b[key])})")
            continue

        a = schema_a[key]
        b = schema_b[key]

        if a.kind != b.kind:
            diffs.append(f"  {key}: kind mismatch: {a.kind} vs {b.kind}")
            continue

        if a.kind == "dataset":
            if a.dtype != b.dtype:
                diffs.append(f"  {key}: dtype mismatch: {a.dtype} vs {b.dtype}")
            if a.shape is not None and b.shape is not None:
                # Compare rank and non-leading dims (leading dim = num timesteps, expected to differ)
                if len(a.shape) != len(b.shape):
                    diffs.append(f"  {key}: rank mismatch: {a.shape} vs {b.shape}")
                elif len(a.shape) > 1 and a.shape[1:] != b.shape[1:]:
                    diffs.append(f"  {key}: trailing shape mismatch: {a.shape} vs {b.shape}")
                elif a.shape[0] != b.shape[0]:
                    diffs.append(f"  {key}: T differs: {a.shape} vs {b.shape} (ok, different episode lengths)")

        # Compare attrs
        attr_keys = sorted(set(a.attrs.keys()) | set(b.attrs.keys()))
        for ak in attr_keys:
            va = a.attrs.get(ak)
            vb = b.attrs.get(ak)
            if va != vb:
                diffs.append(f"  {key} attr '{ak}': {label_a}={va}  vs  {label_b}={vb}")

    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"{'='*70}")
    if not diffs:
        print("  Schemas are identical.")
    else:
        print(f"  Found {len(diffs)} difference(s):\n")
        for d in diffs:
            print(d)
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare two HDF5 file schemas")
    parser.add_argument("--file-a", required=True, help="First HDF5 file")
    parser.add_argument("--file-b", required=True, help="Second HDF5 file")
    args = parser.parse_args()

    with h5py.File(args.file_a, "r") as fa, h5py.File(args.file_b, "r") as fb:
        top_attrs_a = {k: _fmt_attr(v) for k, v in fa.attrs.items()}
        top_attrs_b = {k: _fmt_attr(v) for k, v in fb.attrs.items()}
        schema_a = collect_schema(fa)
        schema_b = collect_schema(fb)

        label_a = args.file_a.split("/")[-1]
        label_b = args.file_b.split("/")[-1]

        print_schema("A", args.file_a, schema_a, top_attrs_a)
        print_schema("B", args.file_b, schema_b, top_attrs_b)
        compare(schema_a, schema_b, top_attrs_a, top_attrs_b, label_a, label_b)


if __name__ == "__main__":
    main()
