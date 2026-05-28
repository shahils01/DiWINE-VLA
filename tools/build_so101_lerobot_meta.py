from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


CAMERA_KEYS = [
    "observation.images.wrist",
    "observation.images.front",
    "observation.images.side",
]


def read_tasks(dataset_root: Path) -> dict[int, str]:
    table = pq.read_table(dataset_root / "meta" / "tasks.parquet").to_pydict()
    return {int(i): str(t) for i, t in zip(table["task_index"], table["task"])}


def build_entries(dataset_root: Path) -> list[dict]:
    tasks = read_tasks(dataset_root)
    episodes_path = dataset_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes = pq.read_table(episodes_path).to_pylist()
    entries = []
    for ep in episodes:
        data_path = dataset_root / "data" / f"chunk-{ep['data/chunk_index']:03d}" / f"file-{ep['data/file_index']:03d}.parquet"
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
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description="Build DiWINE-VLA meta JSON for SO101 LeRobot datasets.")
    parser.add_argument("--dataset_roots", nargs="+", required=True, help="One or more LeRobot dataset roots.")
    parser.add_argument("--output", required=True, help="Output meta JSON path.")
    parser.add_argument("--dataset_name", default="so101", help="DiWINE dataset name.")
    args = parser.parse_args()

    datalist = []
    for root in args.dataset_roots:
        datalist.extend(build_entries(Path(root).resolve()))

    meta = {
        "dataset_name": args.dataset_name,
        "robot_type": "so101",
        "camera_keys": CAMERA_KEYS,
        "datalist": datalist,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {output} with {len(datalist)} episodes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
