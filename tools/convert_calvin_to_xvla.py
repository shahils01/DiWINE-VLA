import argparse
import json
import re
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _load_annotation_payload(path: Path) -> Any:
    if path.suffix == ".npy":
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.shape == ():
            return arr.item()
        return arr
    if path.suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        payload = {k: data[k] for k in data.files}
        if len(payload) == 1:
            only = next(iter(payload.values()))
            if isinstance(only, np.ndarray) and only.shape == ():
                return only.item()
        return payload
    raise ValueError(f"Unsupported annotation file: {path}")


def _to_python(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        if obj.shape == ():
            return _to_python(obj.item())
        return [_to_python(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_python(x) for x in obj]
    return obj


def _extract_segments(payload: Any) -> list[tuple[int, int, str]]:
    payload = _to_python(payload)

    if isinstance(payload, dict):
        if "language" in payload and "info" in payload:
            language = payload["language"]
            info = payload["info"]
            anns = language.get("ann", language) if isinstance(language, dict) else language
            idxs = info.get("indx", info) if isinstance(info, dict) else info
            return _pair_segments(anns, idxs)
        if "ann" in payload and "indx" in payload:
            return _pair_segments(payload["ann"], payload["indx"])
        if "annotations" in payload:
            return _extract_segments(payload["annotations"])

    if isinstance(payload, list):
        segments = []
        for item in payload:
            if isinstance(item, dict):
                text = None
                for key in ("ann", "instruction", "text", "task", "lang"):
                    if isinstance(item.get(key), str) and item[key].strip():
                        text = item[key].strip()
                        break
                if text is None:
                    continue
                if "indx" in item:
                    start, end = item["indx"]
                elif "range" in item:
                    start, end = item["range"]
                elif "start" in item and "end" in item:
                    start, end = item["start"], item["end"]
                else:
                    continue
                segments.append((int(start), int(end), text))
        if segments:
            return segments

    raise ValueError("Could not parse CALVIN language annotations.")


def _pair_segments(anns: Any, idxs: Any) -> list[tuple[int, int, str]]:
    anns = _to_python(anns)
    idxs = _to_python(idxs)
    segments = []
    for ann, idx in zip(anns, idxs):
        if isinstance(ann, list):
            ann = ann[0]
        if not isinstance(ann, str):
            continue
        if isinstance(idx, (list, tuple)) and len(idx) >= 2:
            segments.append((int(idx[0]), int(idx[1]), ann.strip()))
    if not segments:
        raise ValueError("No valid CALVIN language segments found.")
    return segments


def load_language_segments(training_dir: Path) -> list[tuple[int, int, str]]:
    candidates = [
        training_dir / "lang_annotations" / "auto_lang_ann.npy",
        training_dir / "lang_annotations" / "auto_lang_ann.npz",
        training_dir / "lang_annotations.npy",
        training_dir / "lang_annotations.npz",
    ]
    for path in candidates:
        if path.exists():
            return _extract_segments(_load_annotation_payload(path))
    raise FileNotFoundError(
        f"Could not find CALVIN language annotations under {training_dir}. "
        "Expected lang_annotations/auto_lang_ann.npy or similar."
    )


def instruction_for_episode(ep_idx: int, segments: list[tuple[int, int, str]]) -> str:
    for start, end, text in segments:
        if start <= ep_idx < end:
            return text
    for start, end, text in segments:
        if start <= ep_idx <= end:
            return text
    raise KeyError(f"No language annotation found for CALVIN episode index {ep_idx}.")


def convert_training_split(training_dir: Path, out_dir: Path, dataset_name: str, meta_out: Path) -> None:
    data_out = out_dir / "data"
    data_out.mkdir(parents=True, exist_ok=True)

    segments = load_language_segments(training_dir)
    episode_paths = sorted(training_dir.glob("episode_*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {training_dir}.")

    datalist = []
    for src in episode_paths:
        match = re.search(r"episode_(\d+)\.npz$", src.name)
        if not match:
            continue
        ep_idx = int(match.group(1))
        ins = instruction_for_episode(ep_idx, segments)

        data = np.load(src, allow_pickle=True)
        if "robot_obs" not in data or "rgb_static" not in data or "rgb_gripper" not in data:
            raise KeyError(
                f"{src} is missing one of required keys: robot_obs, rgb_static, rgb_gripper."
            )

        robot_obs = data["robot_obs"]
        if robot_obs.shape[-1] < 7:
            raise ValueError(f"{src} robot_obs has shape {robot_obs.shape}; expected at least 7 dims.")
        proprio = robot_obs[:, :7].astype(np.float32)
        rgb_static = data["rgb_static"]
        rgb_gripper = data["rgb_gripper"]

        dst = data_out / f"{src.stem}.hdf5"
        with h5py.File(dst, "w") as f:
            f.create_dataset("proprio", data=proprio, compression="gzip", compression_opts=4)
            f.create_dataset("rgb_static", data=rgb_static, compression="gzip", compression_opts=4)
            f.create_dataset("rgb_gripper", data=rgb_gripper, compression="gzip", compression_opts=4)
            f.create_dataset("language_instruction", data=np.bytes_(ins))

        datalist.append(str(dst.resolve()))

    meta = {
        "dataset_name": dataset_name,
        "robot_type": "Calvin",
        "observation_key": ["rgb_static", "rgb_gripper"],
        "language_instruction_key": "language_instruction",
        "datalist": datalist,
    }

    meta_out.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"wrote {len(datalist)} converted episodes to {data_out}")
    print(f"wrote meta to {meta_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw CALVIN training npz files into X-VLA HDF5 format.")
    parser.add_argument("--training_dir", required=True, help="Path to CALVIN training split, e.g. task_ABC_D/training")
    parser.add_argument("--out_dir", required=True, help="Output directory for converted HDF5 episodes")
    parser.add_argument("--dataset_name", default="Calvin_ABC_D", help="Dataset name stored in meta.json")
    parser.add_argument("--meta_out", required=True, help="Path to output meta.json")
    args = parser.parse_args()

    convert_training_split(
        training_dir=Path(args.training_dir),
        out_dir=Path(args.out_dir),
        dataset_name=args.dataset_name,
        meta_out=Path(args.meta_out),
    )


if __name__ == "__main__":
    main()
