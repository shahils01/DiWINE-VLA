#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _filename_to_instruction(path: Path) -> str:
    stem = path.name.replace("_demo.hdf5", "").replace("_demo", "")
    return stem.replace("_", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build X-VLA meta JSON for LIBERO robomimic HDF5 files.")
    parser.add_argument("--data_root", required=True, help="Folder with LIBERO *.hdf5 files")
    parser.add_argument("--output", required=True, help="Path to write meta json")
    parser.add_argument(
        "--dataset_name",
        default="libero_rmb",
        help="Dataset name to register with X-VLA (default: libero_rmb)",
    )
    parser.add_argument(
        "--observation_key",
        default="obs/agentview_rgb,obs/eye_in_hand_rgb",
        help="Comma-separated observation keys (default: obs/agentview_rgb,obs/eye_in_hand_rgb)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    files = sorted(data_root.glob("*.hdf5"))
    if not files:
        raise FileNotFoundError(f"No .hdf5 files found in {data_root}")

    obs_keys = [k.strip() for k in args.observation_key.split(",") if k.strip()]
    datalist = []
    for fp in files:
        datalist.append(
            {
                "path": str(fp),
                "instruction": _filename_to_instruction(fp),
            }
        )

    meta = {
        "dataset_name": args.dataset_name,
        "robot_type": args.dataset_name,
        "observation_key": obs_keys,
        "datalist": datalist,
    }

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {out_path} with {len(datalist)} files.")


if __name__ == "__main__":
    main()
