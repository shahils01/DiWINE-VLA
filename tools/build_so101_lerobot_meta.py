from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


CAMERA_KEYS = [
    "observation.images.wrist",
    "observation.images.front",
    "observation.images.side",
]


def read_tasks(dataset_root: Path) -> dict[int, str]:
    table = pq.read_table(dataset_root / "meta" / "tasks.parquet").to_pydict()
    return {int(i): str(t) for i, t in zip(table["task_index"], table["task"])}


def build_entries(dataset_root: Path) -> tuple[list[dict], list[Path]]:
    tasks = read_tasks(dataset_root)
    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pq.read_table(episodes_path).to_pylist()
    entries = []
    data_paths = []
    for ep in episodes:
        data_path = dataset_root / "data" / f"chunk-{ep['data/chunk_index']:03d}" / f"file-{ep['data/file_index']:03d}.parquet"
        data_paths.append(data_path)
        video_paths = {}
        for key in CAMERA_KEYS:
            chunk = ep[f"videos/{key}/chunk_index"]
            file_idx = ep[f"videos/{key}/file_index"]
            video_paths[key] = str(dataset_root / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file_idx:03d}.mp4")
        task_list = ep.get("tasks") or []
        task = task_list[0] if task_list else tasks.get(int(ep.get("task_index", 0)), "")
        entries.append(
            {
                "dataset_root": str(dataset_root),
                "episode_index": int(ep["episode_index"]),
                "length": int(ep["length"]),
                "task": task,
                "data_path": str(data_path),
                "video_paths": video_paths,
            }
        )
    return entries, data_paths


def _flatten_column(table: dict, key: str) -> np.ndarray:
    return np.asarray(table[key], dtype=np.float32)


def compute_state_action_stats(data_paths: list[Path]) -> dict:
    actions = []
    states = []
    for data_path in sorted(set(data_paths)):
        table = pq.read_table(data_path, columns=["action", "observation.state"]).to_pydict()
        actions.append(_flatten_column(table, "action"))
        states.append(_flatten_column(table, "observation.state"))
    action = np.concatenate(actions, axis=0)
    state = np.concatenate(states, axis=0)
    both = np.concatenate([action, state], axis=0)
    mean = both.mean(axis=0)
    std = both.std(axis=0)
    std = np.maximum(std, 1e-6)
    return {
        "mode": "mean_std",
        "keys": ["action", "observation.state"],
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "min": both.min(axis=0).astype(float).tolist(),
        "max": both.max(axis=0).astype(float).tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DiWINE-VLA meta JSON for SO101 LeRobot datasets.")
    parser.add_argument("--dataset_roots", nargs="+", required=True, help="One or more LeRobot dataset roots.")
    parser.add_argument("--output", required=True, help="Output meta JSON path.")
    parser.add_argument("--dataset_name", default="so101", help="DiWINE dataset name.")
    args = parser.parse_args()

    datalist = []
    data_paths = []
    for root in args.dataset_roots:
        entries, paths = build_entries(Path(root).resolve())
        datalist.extend(entries)
        data_paths.extend(paths)

    meta = {
        "dataset_name": args.dataset_name,
        "robot_type": "so101",
        "camera_keys": CAMERA_KEYS,
        "normalization": compute_state_action_stats(data_paths),
        "datalist": datalist,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {output} with {len(datalist)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
