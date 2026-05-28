from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from mmengine import fileio
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .domain_config import DATA_DOMAIN_ID, DATA_WEIGHTS
from .utils import action_slice, euler_to_rotate6d


def _load_meta_files(metas_path: str) -> Dict[str, dict]:
    metas: Dict[str, dict] = {}
    if fileio.isdir(metas_path):
        meta_files = fileio.list_dir_or_file(metas_path, suffix=".json", recursive=True, list_dir=False)
        root = metas_path
    else:
        meta_files, root = [metas_path], ""

    for file in meta_files:
        file_path = fileio.join_path(root, file)
        with io.BytesIO(fileio.get(file_path)) as f:
            meta = json.load(f)
        if "dataset_name" in meta and "datalist" in meta:
            metas[meta["dataset_name"]] = meta
        else:
            raise NotImplementedError(f"Unsupported meta file format for RLDS dataset: {file_path}")
    return metas


def contains_rlds_meta(metas_path: str) -> bool:
    metas = _load_meta_files(metas_path)
    is_rlds = [((m.get("robot_type") or "") == "droid_rlds") for m in metas.values()]
    if any(is_rlds) and not all(is_rlds):
        raise NotImplementedError(
            "Mixing native RLDS metas with HDF5 metas in one dataloader is not supported. "
            "Use a pure RLDS meta directory or a pure HDF5 meta directory."
        )
    return all(is_rlds) and len(is_rlds) > 0


def _build_distributed_rank_info() -> tuple[int, int]:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _normalize_image_keys(keys: List[str] | tuple[str, ...] | None) -> List[str]:
    if not keys:
        return ["exterior_image_1_left", "exterior_image_2_left"]
    normalized = []
    for key in keys:
        key = str(key)
        if key.startswith("observation/"):
            key = key.split("/", 1)[1]
        normalized.append(key)
    return normalized


def _resolve_rlds_root(meta: dict) -> str:
    candidates = [
        meta.get("rlds_root"),
        meta.get("root_path"),
    ]
    datalist = meta.get("datalist", [])
    if len(datalist) == 1 and isinstance(datalist[0], str):
        candidates.append(datalist[0])
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / "features.json").exists():
            return str(path)
        if (path / "1.0.0" / "features.json").exists():
            return str(path / "1.0.0")
    raise FileNotFoundError(
        "Could not resolve RLDS dataset root from meta. Provide one of "
        "`rlds_root`, `root_path`, or a single-item `datalist` pointing to the TFDS builder directory."
    )


def _decode_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_text(value.item())
    return str(value)


def _extract_instruction(step: dict, key: str | None = None) -> str:
    if key:
        key = key.split("/")[-1]
        text = _decode_text(step.get(key))
        if text:
            return text
    for fallback in ("language_instruction", "language_instruction_2", "language_instruction_3"):
        text = _decode_text(step.get(fallback))
        if text:
            return text
    return ""


def _extract_view_frames_from_step(step: dict, image_keys: List[str]) -> List[Image.Image]:
    obs = step.get("observation", {})
    frames = []
    for key in image_keys:
        if key in obs:
            image = np.asarray(obs[key])
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            frames.append(Image.fromarray(image))
    if not frames:
        raise RuntimeError(
            f"Requested image keys {image_keys} not present. Available keys: {list(obs.keys())}"
        )
    return frames


def _droid_state_from_step(step: dict) -> np.ndarray:
    obs = step["observation"]
    cart = np.asarray(obs["cartesian_position"], dtype=np.float32)  # xyz + euler_xyz
    grip = np.asarray(obs["gripper_position"], dtype=np.float32).reshape(-1)
    left = np.concatenate(
        [cart[:3], euler_to_rotate6d(cart[3:6][None], "xyz")[0], grip[:1]],
        axis=0,
    ).astype(np.float32)
    right = np.zeros_like(left)
    return np.concatenate([left, right], axis=0)


class InfiniteRLDSReader(IterableDataset):
    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "ee6d",
        future_offset_steps: int = 0,
        future_horizon_steps: int = 1,
        image_aug=None,
        episode_shuffle_buffer: int = 1024,
        shuffle_steps: bool = True,
        max_samples_per_episode: int = 0,
    ):
        self.metas = _load_meta_files(metas_path)
        self.num_actions = int(num_actions)
        self.num_views = int(num_views)
        self.training = bool(training)
        self.action_mode = action_mode
        self.future_offset_steps = int(future_offset_steps)
        self.future_horizon_steps = int(future_horizon_steps)
        if image_aug is None:
            image_aug = [
                transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
                if training else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
            ]
            image_aug = transforms.Compose(image_aug)
        self.image_aug = image_aug
        self.episode_shuffle_buffer = int(episode_shuffle_buffer)
        self.shuffle_steps = bool(shuffle_steps)
        self.max_samples_per_episode = int(max_samples_per_episode)

        invalid = [name for name, meta in self.metas.items() if meta.get("robot_type") != "droid_rlds"]
        if invalid:
            raise NotImplementedError(
                "InfiniteRLDSReader only supports metas with robot_type='droid_rlds'. "
                f"Found incompatible datasets: {invalid}"
            )
        self._builders = None

    def _get_tfds(self):
        try:
            import tensorflow_datasets as tfds
        except Exception as exc:
            raise RuntimeError(
                "Native DROID RLDS loading requires tensorflow-datasets and TensorFlow in the training environment."
            ) from exc
        return tfds

    def _get_builders(self):
        if self._builders is None:
            tfds = self._get_tfds()
            builders = {}
            for name, meta in self.metas.items():
                root = _resolve_rlds_root(meta)
                builders[name] = tfds.builder_from_directory(root)
            self._builders = builders
        return self._builders

    def _iter_dataset(self, dataset_name: str, meta: dict, seed: int) -> Iterable[dict]:
        tfds = self._get_tfds()
        builder = self._get_builders()[dataset_name]
        split = meta.get("rlds_split", "train")
        dataset = builder.as_dataset(split=split)
        if self.training and self.episode_shuffle_buffer > 1:
            dataset = dataset.shuffle(
                buffer_size=self.episode_shuffle_buffer,
                seed=seed,
                reshuffle_each_iteration=False,
            )

        rank, world_size = _build_distributed_rank_info()
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        shard_index = rank * num_workers + worker_id
        shard_count = world_size * num_workers

        rng = np.random.default_rng(seed + 997 * worker_id + 7919 * rank)
        image_keys = _normalize_image_keys(meta.get("observation_key"))
        language_key = meta.get("language_instruction_key", "language_instruction")
        domain_id_key = meta.get("domain_id_key", "droid_rlds")
        domain_id = torch.tensor(
            DATA_DOMAIN_ID.get(domain_id_key, DATA_DOMAIN_ID.get("droid_rlds", 0)),
            dtype=torch.long,
        )

        for episode_idx, episode in enumerate(tfds.as_numpy(dataset)):
            if shard_count > 1 and (episode_idx % shard_count) != shard_index:
                continue

            steps = list(episode.get("steps", []))
            total_steps = len(steps)
            required = max(self.num_actions + 1, self.future_offset_steps + max(1, self.future_horizon_steps))
            max_idx = total_steps - required
            if max_idx < 0:
                continue

            candidate_indices = np.arange(max_idx + 1, dtype=np.int64)
            if self.training and self.shuffle_steps:
                rng.shuffle(candidate_indices)
            if self.max_samples_per_episode > 0 and candidate_indices.size > self.max_samples_per_episode:
                candidate_indices = candidate_indices[: self.max_samples_per_episode]

            image_mask = torch.zeros(self.num_views, dtype=torch.bool)
            image_mask[: min(self.num_views, len(image_keys))] = True
            if not image_mask.any():
                raise RuntimeError("At least one image view is required for RLDS loading.")

            for idx in candidate_indices.tolist():
                instruction = _extract_instruction(steps[idx], language_key)
                trajectory_states = [
                    _droid_state_from_step(steps[int(j)])
                    for j in range(idx, idx + self.num_actions + 1)
                ]
                abs_trajectory = torch.tensor(np.stack(trajectory_states, axis=0), dtype=torch.float32)

                current_frames = _extract_view_frames_from_step(steps[idx], image_keys[: self.num_views])
                image_input = torch.stack([self.image_aug(frame) for frame in current_frames], dim=0)
                while image_input.size(0) < self.num_views:
                    image_input = torch.cat([image_input, torch.zeros_like(image_input[:1])], dim=0)

                sample = {
                    "domain_id": domain_id.clone(),
                    "language_instruction": instruction,
                    "image_input": image_input,
                    "image_mask": image_mask,
                    "abs_trajectory": abs_trajectory,
                }

                if self.future_offset_steps > 0:
                    future_indices = [
                        min(idx + self.future_offset_steps + t, total_steps - 1)
                        for t in range(max(1, self.future_horizon_steps))
                    ]
                    future_frames = []
                    for future_idx in future_indices:
                        future_views = _extract_view_frames_from_step(steps[int(future_idx)], image_keys[: self.num_views])
                        frame_tensor = torch.stack([self.image_aug(frame) for frame in future_views], dim=0)
                        while frame_tensor.size(0) < self.num_views:
                            frame_tensor = torch.cat([frame_tensor, torch.zeros_like(frame_tensor[:1])], dim=0)
                        future_frames.append(frame_tensor)
                    future_proprio = torch.tensor(
                        np.stack([_droid_state_from_step(steps[int(j)]) for j in future_indices], axis=0),
                        dtype=torch.float32,
                    )
                    sample["future_image_input"] = torch.stack(future_frames, dim=0)
                    sample["future_image_mask"] = image_mask
                    sample["future_proprio"] = future_proprio

                sample.update(action_slice(sample.pop("abs_trajectory")))
                yield sample

        if self.training:
            yield from self._iter_dataset(dataset_name, meta, seed + 104729)

    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training:
            for dataset_name in names:
                yield from self._iter_dataset(dataset_name, self.metas[dataset_name], seed=0)
            return

        weights = [DATA_WEIGHTS.get(self.metas[name].get("domain_id_key", "droid_rlds"), DATA_WEIGHTS.get("droid_rlds", 1.0)) for name in names]
        total = sum(weights)
        weights = [w / total for w in weights]
        gens = [iter(self._iter_dataset(name, self.metas[name], seed=1000 * i + 17)) for i, name in enumerate(names)]
        while True:
            i = random.choices(range(len(names)), weights=weights, k=1)[0]
            yield next(gens[i])
