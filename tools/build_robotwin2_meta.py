import argparse
import json
from pathlib import Path


def infer_robot_type(embodiment: str) -> str:
    stem = embodiment
    for suffix in ("_clean_50", "_randomized_500"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    mapping = {
        "aloha-agilex": "robotwin2-aloha-agilex",
        "arx-x5": "robotwin2-arx-x5",
        "franka": "robotwin2-franka",
        "ur5": "robotwin2-ur5",
    }
    if stem in mapping:
        return mapping[stem]
    raise ValueError(
        f"Could not infer robot_type from embodiment '{embodiment}'. "
        "Pass --robot_type explicitly."
    )


def build_meta(dataset_root: Path, embodiment: str, output: Path, dataset_name: str, robot_type: str) -> None:
    datalist = []
    for task_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        split_dir = task_dir / embodiment
        data_dir = split_dir / "data"
        if not data_dir.is_dir():
            continue
        datalist.extend(str(p.resolve()) for p in sorted(data_dir.glob("*.hdf5")))

    if not datalist:
        raise FileNotFoundError(
            f"No HDF5 episodes found under {dataset_root} for embodiment split '{embodiment}'."
        )

    meta = {
        "dataset_name": dataset_name,
        "robot_type": robot_type,
        "observation_key": [
            "observation/head_camera/rgb",
            "observation/left_camera/rgb",
            "observation/right_camera/rgb",
        ],
        "language_instruction_key": "sidecar_json",
        "datalist": datalist,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"wrote {output} with {len(datalist)} episodes")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build X-VLA meta.json for RoboTwin2.")
    parser.add_argument("--dataset_root", required=True, help="Path to RoboTwin2 dataset root containing task folders.")
    parser.add_argument("--embodiment", required=True, help="Embodiment split folder name, e.g. ur5_clean_50.")
    parser.add_argument("--output", required=True, help="Path to output meta.json.")
    parser.add_argument("--dataset_name", default="robotwin2", help="Dataset name stored in meta.json.")
    parser.add_argument(
        "--robot_type",
        default=None,
        help="Robot type for X-VLA handler registry. If omitted, infer from --embodiment.",
    )
    args = parser.parse_args()
    robot_type = args.robot_type or infer_robot_type(args.embodiment)
    dataset_name = args.dataset_name
    if dataset_name == "robotwin2":
        dataset_name = robot_type

    build_meta(
        dataset_root=Path(args.dataset_root),
        embodiment=args.embodiment,
        output=Path(args.output),
        dataset_name=dataset_name,
        robot_type=robot_type,
    )


if __name__ == "__main__":
    main()
