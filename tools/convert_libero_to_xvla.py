#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import h5py
import numpy as np
from tqdm import tqdm


def axis_angle_to_rotmat_batch(aa: np.ndarray) -> np.ndarray:
    aa = aa.astype(np.float64)
    T = aa.shape[0]
    theta = np.linalg.norm(aa, axis=1, keepdims=True)
    axis = aa / (theta + 1e-12)

    ax, ay, az = axis[:, 0], axis[:, 1], axis[:, 2]
    zeros = np.zeros(T, dtype=np.float64)
    K = np.stack(
        [zeros, -az, ay, az, zeros, -ax, -ay, ax, zeros],
        axis=1,
    ).reshape(T, 3, 3)

    I = np.eye(3, dtype=np.float64)[None, :, :]
    sin = np.sin(theta)[:, None]
    cos = np.cos(theta)[:, None]
    K2 = K @ K
    R = I + sin * K + (1.0 - cos) * K2

    small = theta[:, 0] < 1e-8
    if np.any(small):
        R[small] = np.eye(3, dtype=np.float64)
    return R


def rotmat_to_rot6d_batch(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def parse_language_from_filename(stem: str) -> str:
    s = re.sub(r"_demo$", "", stem)
    m = re.match(r"^.*?_SCENE\\d+_(.+)$", s)
    if m:
        return m.group(1)
    return s


def gripper_states_to_grip_raw(gripper_states: np.ndarray) -> np.ndarray:
    g = np.asarray(gripper_states)
    if g.ndim == 1:
        g = g[:, None]
    opening = g.mean(axis=1)
    closed = opening < 0.02
    return np.where(closed, 1.0, -1.0).astype(np.float32)[:, None]


def _flip_agentview(img: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(img, 0), 1)


def convert_dir(
    src_dir: Path,
    out_dir: Path,
    dataset_name: str,
    meta_json_path: Path,
    max_episodes: int | None = None,
) -> None:
    data_dir = out_dir / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    datalist: List[str] = []
    ep_count = 0
    h5_files = sorted(src_dir.glob("*.hdf5"))
    if not h5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {src_dir}")

    for idx, src_path in enumerate(h5_files):
        print(f"File {idx + 1}/{len(h5_files)}: {src_path.name}")
        stem = src_path.stem
        lang_default = parse_language_from_filename(stem)

        with h5py.File(src_path, "r") as f:
            demos = sorted(list(f["data"].keys()))
            for demo in tqdm(demos, desc="demos"):
                grp = f[f"data/{demo}"]
                obs = grp["obs"]

                agent = obs["agentview_rgb"][()]
                eih = obs["eye_in_hand_rgb"][()]
                agent = np.stack([_flip_agentview(img) for img in agent], axis=0)

                ee_pos = obs["ee_pos"][()].astype(np.float32)
                ee_ori = obs["ee_ori"][()].astype(np.float64)
                R = axis_angle_to_rotmat_batch(ee_ori)
                rot6d = rotmat_to_rot6d_batch(R).astype(np.float32)
                grip_raw = gripper_states_to_grip_raw(obs["gripper_states"][()])

                abs_action_6d = np.concatenate([ee_pos, rot6d, grip_raw], axis=1)
                assert abs_action_6d.shape[1] == 10

                out_path = data_dir / f"episode_{ep_count:06d}.hdf5"
                with h5py.File(out_path, "w") as fo:
                    fo.create_dataset("abs_action_6d", data=abs_action_6d, compression="gzip", compression_opts=4)
                    fo.create_dataset("language_instruction", data=np.string_(lang_default))
                    fo.create_dataset("agentview_rgb", data=agent, compression="gzip", compression_opts=4)
                    fo.create_dataset("eye_in_hand_rgb", data=eih, compression="gzip", compression_opts=4)

                datalist.append(str(out_path.resolve()))
                ep_count += 1
                if max_episodes is not None and ep_count >= max_episodes:
                    break
            if max_episodes is not None and ep_count >= max_episodes:
                break

    meta = {
        "dataset_name": dataset_name,
        "robot_type": "libero",
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
        "datalist": datalist,
    }
    meta_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_json_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"Done. episodes={ep_count}, meta={meta_json_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", required=True, help="LIBERO dataset dir with *.hdf5")
    parser.add_argument("--out_dir", required=True, help="Output directory for XVLA-format dataset")
    parser.add_argument("--dataset_name", required=True, help="Dataset name in meta.json")
    parser.add_argument("--meta_out", required=True, help="Path to meta.json")
    parser.add_argument("--max_episodes", type=int, default=None, help="Optional cap for quick tests")
    args = parser.parse_args()

    convert_dir(
        src_dir=Path(args.src_dir),
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
        meta_json_path=Path(args.meta_out),
        max_episodes=args.max_episodes,
    )


if __name__ == "__main__":
    main()
