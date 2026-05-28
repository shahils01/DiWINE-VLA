import json
from pathlib import Path

root = Path("/scratch/shahils/openpi/datasets/Libero-XVLA-format")
files = sorted(root.glob("libero_*/*_demo/demo_*.hdf5"))

meta = {
      "dataset_name": "libero",
      "robot_type": "libero",
      "datalist": [str(p) for p in files],
      "observation_key": ["observation/third_image", "observation/wrist_image"],
      "language_instruction_key": "language_instruction",
}

out = root / f"libero_meta.json"
out.write_text(json.dumps(meta, indent=2))
print(f"Wrote {len(files)} demos to {out}")


