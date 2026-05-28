from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from ..utils import read_parquet, read_video_to_frames
from .base import DomainHandler


class SO101LeRobotHandler(DomainHandler):
    """Current LeRobot parquet/video handler for SO101 datasets.

    Expected meta entries are produced by tools/build_so101_lerobot_meta.py.
    Actions and proprio are SO101 joint/gripper positions [T, 6]. They are padded
    to 20D so DiWINE/X-VLA can use action_mode=auto with real_action_dim=6.
    """

    CAMERA_KEYS = [
        "observation.images.wrist",
        "observation.images.front",
        "observation.images.side",
    ]
    REAL_ACTION_DIM = 6
    MODEL_ACTION_DIM = 20

    def _normalization_stats(self) -> tuple[np.ndarray, np.ndarray]:
        stats = self.meta.get("normalization", {})
        if stats.get("mode") != "mean_std":
            mean = np.zeros(self.REAL_ACTION_DIM, dtype=np.float32)
            std = np.ones(self.REAL_ACTION_DIM, dtype=np.float32)
            return mean, std
        mean = np.asarray(stats["mean"], dtype=np.float32)[: self.REAL_ACTION_DIM]
        std = np.asarray(stats["std"], dtype=np.float32)[: self.REAL_ACTION_DIM]
        return mean, np.maximum(std, 1e-6)

    def _normalize_real_dims(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        mean, std = self._normalization_stats()
        x = x[..., : self.REAL_ACTION_DIM]
        return (x - mean) / std

    def _pad20(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] >= self.MODEL_ACTION_DIM:
            return x[..., : self.MODEL_ACTION_DIM]
        pad = np.zeros((*x.shape[:-1], self.MODEL_ACTION_DIM - x.shape[-1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=-1)

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
        camera_keys = self.meta.get("camera_keys", self.CAMERA_KEYS)
        images = [read_video_to_frames(item["video_paths"][key]) for key in camera_keys]
        data = read_parquet(item["data_path"])

        actions = self._pad20(self._normalize_real_dims(np.asarray(data["action"], dtype=np.float32)))
        states = self._pad20(self._normalize_real_dims(np.asarray(data["observation.state"], dtype=np.float32)))
        T = min([len(actions), len(states), *(len(img) for img in images)])
        if T <= num_actions + 1:
            return

        actions = actions[:T]
        states = states[:T]
        images = [img[:T] for img in images]

        instruction = item.get("task") or item.get("tasks", [""])[0]
        future_offset_steps = int(kwargs.get("future_offset_steps", 0) or 0)
        future_horizon_steps = int(kwargs.get("future_horizon_steps", 1) or 1)
        max_start = T - (num_actions + 1)
        if future_offset_steps > 0:
            max_start = min(max_start, T - (future_offset_steps + future_horizon_steps))
        if max_start <= 0:
            return

        idxs = list(range(max_start))
        if training:
            random.shuffle(idxs)
        image_mask = torch.ones(self.num_views, dtype=torch.bool)
        if training and lang_aug_map and instruction in lang_aug_map:
            instruction = random.choice(lang_aug_map[instruction])

        for idx in idxs:
            imgs = [
                image_aug(Image.fromarray(images[v][idx]))
                for v in range(min(self.num_views, len(images)))
            ]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            # Make first row the current proprio, following rows future actions.
            abs_traj = np.concatenate([states[idx : idx + 1], actions[idx + 1 : idx + num_actions + 1]], axis=0)
            sample = {
                "language_instruction": instruction,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": torch.tensor(abs_traj, dtype=torch.float32),
            }

            if future_offset_steps > 0:
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
                    sample["future_image_mask"] = image_mask.unsqueeze(0).expand(future_horizon_steps, -1)
                    sample["future_proprio"] = torch.tensor(states[start:end], dtype=torch.float32)

            yield sample
