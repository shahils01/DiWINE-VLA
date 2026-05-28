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

import torch
from torch.utils.data import DataLoader
from .dataset import InfiniteDataReader
from .rlds_dataset import InfiniteRLDSReader, contains_rlds_meta

def worker_init_fn(worker_id: int):
    base_seed = torch.initial_seed() % (2**32)
    import random, numpy as np
    np.random.seed(base_seed); random.seed(base_seed); torch.manual_seed(base_seed)


def create_dataloader(batch_size: int, 
                      metas_path: str, 
                      num_actions: int,
                      training: bool,
                      action_mode: str,
                      num_views: int = 3,
                      num_workers: int = 4,
                      future_offset_steps: int = 0,
                      future_horizon_steps: int = 1,
                      rlds_episode_shuffle_buffer: int = 1024,
                      rlds_shuffle_steps: bool = True,
                      rlds_max_samples_per_episode: int = 0,
                      ):
    if contains_rlds_meta(metas_path):
        dataset = InfiniteRLDSReader(
            metas_path,
            num_actions=num_actions,
            num_views=num_views,
            training=training,
            action_mode=action_mode,
            future_offset_steps=future_offset_steps,
            future_horizon_steps=future_horizon_steps,
            episode_shuffle_buffer=rlds_episode_shuffle_buffer,
            shuffle_steps=rlds_shuffle_steps,
            max_samples_per_episode=rlds_max_samples_per_episode,
        )
    else:
        dataset = InfiniteDataReader(
            metas_path,
            num_actions=num_actions,
            num_views=num_views,
            training=training,
            action_mode=action_mode,
            future_offset_steps=future_offset_steps,
            future_horizon_steps=future_horizon_steps,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=bool(num_workers > 0)
    )
