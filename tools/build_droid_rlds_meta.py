from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser("Build X-VLA native DROID RLDS meta")
    parser.add_argument("--rlds_root", type=str, required=True, help="Path to TFDS builder directory or its parent containing 1.0.0/")
    parser.add_argument("--output", type=str, required=True, help="Output meta.json path")
    parser.add_argument("--dataset_name", type=str, default="droid_rlds")
    parser.add_argument("--rlds_split", type=str, default="train")
    parser.add_argument("--domain_id_key", type=str, default="droid_rlds")
    parser.add_argument("--image_keys", type=str, default="exterior_image_1_left,exterior_image_2_left")
    parser.add_argument("--language_instruction_key", type=str, default="language_instruction")
    args = parser.parse_args()

    image_keys = [x.strip() for x in args.image_keys.split(",") if x.strip()]
    meta = {
        "dataset_name": args.dataset_name,
        "robot_type": "droid_rlds",
        "domain_id_key": args.domain_id_key,
        "rlds_root": args.rlds_root,
        "rlds_split": args.rlds_split,
        "observation_key": image_keys,
        "language_instruction_key": args.language_instruction_key,
        "datalist": [args.rlds_root],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
