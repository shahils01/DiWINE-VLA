# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

from typing import Iterable, Sequence
import os
import random
import numpy as np
import h5py
import torch
from PIL import Image

from ..utils import euler_to_rotate6d
from .base import DomainHandler, _open_h5


class LiberoRoboMimicHandler(DomainHandler):
    """
    LIBERO (robomimic-style HDF5) handler.

    Expected HDF5 layout per file:
      data/demo_x/obs/agentview_rgb   [T, H, W, 3]
      data/demo_x/obs/eye_in_hand_rgb [T, H, W, 3]
      data/demo_x/obs/ee_pos          [T, 3]
      data/demo_x/obs/ee_ori          [T, 3] (euler xyz)
      data/demo_x/obs/gripper_states  [T, 2] (use first channel)
    """

    dataset_name = "libero_rmb"

    def _get_demo_names(self, f: h5py.File) -> list[str]:
        if "data" not in f:
            raise ValueError("LIBERO HDF5 missing top-level 'data' group.")
        return sorted([k for k in f["data"].keys() if k.startswith("demo_")])

    def _get_images(self, f: h5py.File, demo: str, keys: Sequence[str]) -> list[np.ndarray]:
        images = []
        for key in keys:
            images.append(f[f"data/{demo}/{key}"][()])
        return images

    def _get_instruction(self, item, path: str) -> str:
        if isinstance(item, dict) and "instruction" in item:
            return str(item["instruction"])
        stem = os.path.basename(path).replace("_demo.hdf5", "").replace("_demo", "")
        return stem.replace("_", " ")

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        action_mode,
        **kwargs,
    ) -> Iterable[dict]:
        item = self.meta["datalist"][traj_idx]
        path = item["path"] if isinstance(item, dict) else item
        instruction = self._get_instruction(item, path)
        obs_keys = self.meta.get("observation_key", ["obs/agentview_rgb", "obs/eye_in_hand_rgb"])
        future_offset_steps = int(kwargs.get("future_offset_steps", 0) or 0)
        future_horizon_steps = int(kwargs.get("future_horizon_steps", 1) or 1)

        with _open_h5(path) as f:
            demo_names = self._get_demo_names(f)
            for demo in demo_names:
                images = self._get_images(f, demo, obs_keys)
                ee_pos = f[f"data/{demo}/obs/ee_pos"][()]
                ee_ori = f[f"data/{demo}/obs/ee_ori"][()]
                grip = f[f"data/{demo}/obs/gripper_states"][()]
                if grip.ndim == 2:
                    grip = grip[:, :1]
                else:
                    grip = grip[:, None]
                rot6d = euler_to_rotate6d(ee_ori, "xyz")
                left = np.concatenate([ee_pos, rot6d, grip.astype(np.float32)], axis=-1)  # [T,10]
                right = np.zeros_like(left)
                abs_traj = torch.tensor(np.concatenate([left, right], axis=-1), dtype=torch.float32)

                T = abs_traj.shape[0]
                max_start = T - (num_actions + 1)
                max_future_end = T - (future_offset_steps + future_horizon_steps)
                max_start = min(max_start, max_future_end)
                if max_start <= 0:
                    continue
                idxs = list(range(max_start))
                if training:
                    random.shuffle(idxs)
                if training and lang_aug_map and instruction in lang_aug_map:
                    instruction_use = random.choice(lang_aug_map[instruction])
                else:
                    instruction_use = instruction

                for idx in idxs:
                    imgs = [
                        image_aug(Image.fromarray(images[v][idx]))
                        for v in range(min(self.num_views, len(images)))
                    ]
                    while len(imgs) < self.num_views:
                        imgs.append(torch.zeros_like(imgs[0]))
                    image_input = torch.stack(imgs, dim=0)

                    sample = {
                        "language_instruction": instruction_use,
                        "image_input": image_input,
                        "image_mask": torch.ones(self.num_views, dtype=torch.bool),
                        "abs_trajectory": abs_traj[idx : idx + num_actions + 1],
                    }
                    if future_offset_steps >= 0 and future_horizon_steps > 0:
                        start = idx + future_offset_steps
                        end = start + future_horizon_steps
                        if end <= T:
                            future_imgs_seq = []
                            for t in range(start, end):
                                future_imgs = [
                                    image_aug(Image.fromarray(images[v][t]))
                                    for v in range(min(self.num_views, len(images)))
                                ]
                                while len(future_imgs) < self.num_views:
                                    future_imgs.append(torch.zeros_like(future_imgs[0]))
                                future_imgs_seq.append(torch.stack(future_imgs, dim=0))
                            sample["future_image_input"] = torch.stack(future_imgs_seq, dim=0)
                            sample["future_image_mask"] = sample["image_mask"].unsqueeze(0).expand(future_horizon_steps, -1)
                            sample["future_proprio"] = abs_traj[start:end]
                    yield sample
