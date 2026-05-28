#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

import h5py
import numpy as np

from datasets.utils import euler_to_rotate6d


def _first_demo_name(f: h5py.File) -> str:
    if "data" not in f:
        raise ValueError("Missing 'data' group in HDF5.")
    demos = sorted([k for k in f["data"].keys() if k.startswith("demo_")])
    if not demos:
        raise ValueError("No demo_* groups found.")
    return demos[0]


def _load_first_path(meta: dict) -> str:
    dl = meta.get("datalist", [])
    if not dl:
        raise ValueError("Meta has empty datalist.")
    item: Any = dl[0]
    if isinstance(item, dict):
        return item["path"]
    return str(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", required=True, help="Path to X-VLA meta.json")
    args = parser.parse_args()

    with open(args.meta, "r") as f:
        meta = json.load(f)

    path = _load_first_path(meta)
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as f:
        demo = _first_demo_name(f)
        ee_pos = f[f"data/{demo}/obs/ee_pos"][()]
        ee_ori = f[f"data/{demo}/obs/ee_ori"][()]
        grip = f[f"data/{demo}/obs/gripper_states"][()]
        if grip.ndim == 2:
            grip = grip[:, :1]
        else:
            grip = grip[:, None]
        rot6d = euler_to_rotate6d(ee_ori, "xyz")
        left = np.concatenate([ee_pos, rot6d, grip.astype(np.float32)], axis=-1)
        right = np.zeros_like(left)
        abs_traj = np.concatenate([left, right], axis=-1)

    print("Meta dataset_name:", meta.get("dataset_name"))
    print("Robot type:", meta.get("robot_type"))
    print("Observation keys:", meta.get("observation_key"))
    print("Example file:", path)
    print("Demo:", demo)
    print("Left arm shape:", left.shape)
    print("Abs trajectory shape:", abs_traj.shape)
    print("Per-step action dim:", abs_traj.shape[-1])
    print("Left step dims:", left.shape[-1])


if __name__ == "__main__":
    main()
