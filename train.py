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

import os
import math
import time
import json
import random
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

import logging
import os
import sys
import psutil

# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")
    parser.add_argument("--action_mode", type=str, default=None, help="Override pretrained config action_mode.")
    parser.add_argument("--real_action_dim", type=int, default=None, help="Override real action dim for action_mode=auto.")
    parser.add_argument("--max_action_dim", type=int, default=None, help="Override max action dim for action_mode=auto.")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_views", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--rlds_episode_shuffle_buffer", type=int, default=1024)
    parser.add_argument("--rlds_max_samples_per_episode", type=int, default=0)
    parser.add_argument("--disable_rlds_step_shuffle", action="store_true", default=False)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=600000)
    parser.add_argument("--freeze_steps", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--train_vlm_pre", action="store_true", default=False)
    parser.add_argument("--vlm_pre_train_start_step", type=int, default=-1)
    parser.add_argument("--vlm_pre_train_steps", type=int, default=-1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)

    # System
    parser.add_argument("--seed", type=int, default=0)

    # Future prediction (optional)
    parser.add_argument("--use_future_prediction", action="store_true", default=False)
    parser.add_argument("--future_offset_steps", type=int, default=1)
    parser.add_argument("--predictor_hidden_dim", type=int, default=1024)
    parser.add_argument("--num_future_tokens", type=int, default=4)
    parser.add_argument("--num_future_samples", type=int, default=4)
    parser.add_argument("--flow_sampling_steps", type=int, default=16)
    parser.add_argument("--future_loss_weight", type=float, default=1.0)
    parser.add_argument("--future_nll_weight", type=float, default=1.0)
    parser.add_argument("--future_kl_weight", type=float, default=1.0)
    parser.add_argument("--future_horizon_steps", type=int, default=1)
    parser.add_argument("--future_latent_dim", type=int, default=256)
    parser.add_argument("--future_hidden_dim", type=int, default=512)
    parser.add_argument("--future_inject_layer_idx", type=int, default=-1)
    parser.add_argument("--use_future_contrastive", action="store_true", default=False)
    parser.add_argument("--contrastive_loss_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_num_layers", type=int, default=3)
    parser.add_argument("--contrastive_proj_dim", type=int, default=256)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Split param groups by module type with different learning rates."""
    vlm_pre_params, vlm_post_params = model.get_vlm_param_groups()
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    future_params = []
    if getattr(model, "future_compressor", None) is not None:
        future_params.extend(list(model.future_compressor.parameters()))
    if getattr(model, "future_predictor", None) is not None:
        future_params.extend(list(model.future_predictor.parameters()))
    if getattr(model, "future_projector", None) is not None:
        future_params.extend(list(model.future_projector.parameters()))
    if getattr(model, "future_state_proj", None) is not None:
        future_params.extend(list(model.future_state_proj.parameters()))
    if getattr(model, "contrastive_projector", None) is not None:
        future_params.extend(list(model.contrastive_projector.parameters()))
    exclude = set(map(id, vlm_pre_params + vlm_post_params + soft_prompt_params + action_params + future_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm_pre", "params": vlm_pre_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "vlm_post", "params": vlm_post_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
        {"name": "future_heads", "params": future_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups: 
        if g["name"] == name: g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name: return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start: return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Elegant group-wise LR scheduler."""
    base = {
        "vlm_pre": args.learning_rate * args.learning_coef,
        "vlm_post": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
        "future_heads": args.learning_rate,
    }
    def schedule(step, base_lr):
        return linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio)

    def should_train_vlm_pre() -> bool:
        if not args.train_vlm_pre:
            return False
        start = args.vlm_pre_train_start_step if args.vlm_pre_train_start_step >= 0 else args.freeze_steps
        if step < start:
            return False
        if args.vlm_pre_train_steps >= 0 and step >= start + args.vlm_pre_train_steps:
            return False
        return True

    if step < args.freeze_steps:
        set_group_lr(optim, "vlm_pre", 0.0)
        set_group_lr(optim, "vlm_post", base["vlm_post"])
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
        set_group_lr(optim, "future_heads", base["future_heads"])
    else:
        for name, base_lr in base.items():
            if name == "vlm_pre" and not should_train_vlm_pre():
                set_group_lr(optim, name, 0.0)
                continue
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    accelerator = Accelerator(
        log_with="tensorboard", 
        project_dir=output_dir
    )
    accelerator.init_trackers("XVLA-Training")
    
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")

    # Load model & processor
    model_kwargs = {}
    if args.action_mode is not None:
        model_kwargs["action_mode"] = args.action_mode
    if args.real_action_dim is not None:
        model_kwargs["real_action_dim"] = args.real_action_dim
    if args.max_action_dim is not None:
        model_kwargs["max_action_dim"] = args.max_action_dim
    model = XVLA.from_pretrained(args.models, **model_kwargs)
    processor = XVLAProcessor.from_pretrained(args.models)

    # Iterable dataloader (don't wrap with prepare)
    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        num_views=args.num_views,
        num_workers=args.num_workers,
        future_offset_steps=args.future_offset_steps,
        future_horizon_steps=args.future_horizon_steps,
        rlds_episode_shuffle_buffer=args.rlds_episode_shuffle_buffer,
        rlds_shuffle_steps=not args.disable_rlds_step_shuffle,
        rlds_max_samples_per_episode=args.rlds_max_samples_per_episode,
    )

    if args.use_future_prediction:
        model.configure_future_prediction(
            enabled=True,
            predictor_hidden_dim=args.predictor_hidden_dim,
            num_future_tokens=args.num_future_tokens,
            num_future_samples=args.num_future_samples,
            flow_sampling_steps=args.flow_sampling_steps,
            future_loss_weight=args.future_loss_weight,
            future_horizon_steps=args.future_horizon_steps,
            future_nll_weight=args.future_nll_weight,
            future_kl_weight=args.future_kl_weight,
            future_latent_dim=args.future_latent_dim,
            future_hidden_dim=args.future_hidden_dim,
            future_inject_layer_idx=args.future_inject_layer_idx,
            use_future_contrastive=args.use_future_contrastive,
            contrastive_loss_weight=args.contrastive_loss_weight,
            contrastive_num_layers=args.contrastive_num_layers,
            contrastive_proj_dim=args.contrastive_proj_dim,
            contrastive_temperature=args.contrastive_temperature,
        )

    # Optimizer
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optim = accelerator.prepare(model, optim)

    # Training loop
    model.train()
    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start training for {args.iters} iterations | world_size={accelerator.num_processes}")
    
    for batch in train_dataloader:
        # Encode language
        lang = processor.encode_language(batch["language_instruction"])
        batch.pop("language_instruction", None)
        inputs = {**batch, **lang}
        inputs = {k: v.cuda(non_blocking=True) for k, v in inputs.items()}
        # Update LR per group
        update_group_lrs(optim, global_step, args)

        # Forward & backward
        loss_dict: Dict[str, torch.Tensor] = model(**inputs)
        loss = sum(loss_dict.values())
        accelerator.backward(loss)
        if args.max_grad_norm:
            accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optim.step()
        optim.zero_grad()

        # Logging
        if global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            logs["loss_total"] = float(loss.detach().item())
            action_keys = [
                k for k in logs
                if k.endswith("_loss") and k not in ("future_nll_loss", "future_kl_loss", "future_contrastive_loss")
            ]
            logs["loss_action_total"] = sum(logs[k] for k in action_keys) if action_keys else 0.0
            logs["loss_rssm_total"] = (
                logs.get("future_nll_loss", 0.0)
                + logs.get("future_kl_loss", 0.0)
                + logs.get("future_contrastive_loss", 0.0)
            )
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                gpu_mem = torch.cuda.memory_allocated() / 1024**3
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"action={logs['loss_action_total']:.4f} "
                    f"rssm={logs['loss_rssm_total']:.4f} "
                    f"nll={logs.get('future_nll_loss', 0.0):.4f} "
                    f"kl={logs.get('future_kl_loss', 0.0):.4f} "
                    f"ctr={logs.get('future_contrastive_loss', 0.0):.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_vlm_pre={logs['lr_vlm_pre']:.2e} "
                    f"lr_vlm_post={logs['lr_vlm_post']:.2e} ({dt:.2f}s/it) "
                    f"USED_CPU={cpu_mem:.2e} GB "
                    f"USED_GPU={gpu_mem:.2e} GB "
                )
        
        # Checkpointing
        global_step += 1
        if accelerator.is_main_process:
            if global_step == args.iters or global_step % args.save_interval == 0:
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                processor.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step}, f)
        if global_step >= args.iters: break

    accelerator.end_training()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
