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

import logging
import traceback
from typing import Any, Dict

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import uvicorn
import json_numpy
import cv2

from transformers import PreTrainedModel
from .modeling_florence2 import (
    Florence2ForConditionalGeneration,
    _prepare_4d_attention_mask,
    _prepare_4d_attention_mask_for_sdpa,
)
from .transformer import SoftPromptedTransformer
from .action_hub import build_action_space
from .configuration_xvla import XVLAConfig


class RSSMFuturePredictor(torch.nn.Module):
    def __init__(self, cond_dim: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.cond_proj = torch.nn.Linear(cond_dim, hidden_dim)
        self.prior_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.post_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + cond_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.gru = torch.nn.GRUCell(latent_dim, hidden_dim)
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, 2 * cond_dim),
        )

    @staticmethod
    def _split(mu_logvar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def _gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor, mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        return 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)

    def _decode(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu_logvar = self.decoder(h)
        return self._split(mu_logvar)

    def elbo(
        self,
        cond: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        cond: [B, D]
        targets: [B, H, D]
        """
        cond = cond.float()
        targets = targets.float()
        B, H, D = targets.shape
        h = torch.tanh(self.cond_proj(cond))
        recon_losses = []
        kl_losses = []
        for t in range(H):
            prior_mu, prior_logvar = self._split(self.prior_net(h))
            post_mu, post_logvar = self._split(self.post_net(torch.cat([h, targets[:, t]], dim=-1)))
            eps = torch.randn_like(post_mu)
            z = post_mu + eps * torch.exp(0.5 * post_logvar)
            # GRUCell BF16 fused kernel is not available on this stack; keep recurrence in FP32.
            with torch.autocast(device_type="cuda", enabled=False):
                h = self.gru(z.float(), h.float())
            pred_mu, pred_logvar = self._decode(h)
            var = torch.exp(pred_logvar)
            log2pi = torch.log(torch.tensor(2.0 * np.pi, device=var.device, dtype=var.dtype))
            nll = 0.5 * (((targets[:, t] - pred_mu) ** 2) / var + pred_logvar + log2pi)
            recon_losses.append(nll.mean())
            kl = self._gaussian_kl(post_mu, post_logvar, prior_mu, prior_logvar)
            kl_losses.append(kl.mean())
        return torch.stack(recon_losses).mean(), torch.stack(kl_losses).mean()

    @torch.no_grad()
    def rollout(self, cond: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns predicted mean/std for each step.
        """
        cond = cond.float()
        h = torch.tanh(self.cond_proj(cond))
        means = []
        stds = []
        for _ in range(int(horizon)):
            prior_mu, prior_logvar = self._split(self.prior_net(h))
            z = prior_mu
            with torch.autocast(device_type="cuda", enabled=False):
                h = self.gru(z.float(), h.float())
            pred_mu, pred_logvar = self._decode(h)
            means.append(pred_mu)
            stds.append(torch.exp(0.5 * pred_logvar))
        mean = torch.stack(means, dim=1)
        std = torch.stack(stds, dim=1)
        return mean, std

    @torch.no_grad()
    def sample_rollout(self, cond: torch.Tensor, horizon: int, num_samples: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns sampled trajectories plus predictive mean/std.
        samples: [B, S, H, D]
        mean: [B, H, D]
        std: [B, H, D]
        """
        mean, std = self.rollout(cond, horizon)
        eps = torch.randn(
            mean.size(0),
            int(num_samples),
            mean.size(1),
            mean.size(2),
            device=mean.device,
            dtype=mean.dtype,
        )
        samples = mean.unsqueeze(1) + eps * std.unsqueeze(1)
        return samples, mean, std


class FutureTokenCompressor(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_tokens: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.queries = torch.nn.Parameter(torch.randn(num_tokens, hidden_dim) * 0.02)
        self.attn = torch.nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        batch = seq.size(0)
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        tokens, _ = self.attn(queries, seq, seq, need_weights=False)
        return self.norm(tokens)


class FutureTokenProjector(torch.nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = torch.nn.LayerNorm(hidden_dim)

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(stats))


class ContrastiveProjector(torch.nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class XVLA(PreTrainedModel):
    """
    XVLA: HuggingFace-compatible Vision-Language-Action policy.

    Components:
      • Florence2 encoder-only backbone (vision-language)
      • SoftPromptedTransformer (temporal/action head)
      • Action space (pre/post-processing + loss)
    """
    config_class = XVLAConfig
    base_model_prefix = "xvla"
    supports_gradient_checkpointing = True

    def __init__(self, config: XVLAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        # Core settings
        self.num_actions: int = config.num_actions
        self.use_proprio: bool = config.use_proprio
        self.action_mode: str = config.action_mode.lower()
        self.inference_normalization: dict | None = None
        # Action space (dimensions + hooks)
        if config.action_mode.lower() == "auto":
            self.action_space = build_action_space(
                config.action_mode.lower(),
                real_dim=config.real_action_dim,
                max_dim=config.max_action_dim,
            )
        else:
            self.action_space = build_action_space(config.action_mode.lower())
        dim_action = self.action_space.dim_action
        dim_proprio = getattr(self.action_space, "dim_proprio", dim_action)
        self.dim_proprio = dim_proprio

        # Florence2 backbone (encoder only)
        self.vlm = Florence2ForConditionalGeneration(config.florence_config).to(torch.float32)
        if hasattr(self.vlm, "language_model"):
            lm = self.vlm.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "decoder"):
                del lm.model.decoder
            if hasattr(lm, "lm_head"):
                del lm.lm_head

        projection_dim = getattr(self.vlm.config, "projection_dim", None)
        if projection_dim is None:
            raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")

        # Temporal/action head
        self.transformer = SoftPromptedTransformer(
            hidden_size=config.hidden_size,
            multi_modal_input_size=projection_dim,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            num_domains=config.num_domains,
            dim_action=dim_action,
            dim_propio=dim_proprio,
            len_soft_prompts=config.len_soft_prompts,
            dim_time=config.dim_time,
            max_len_seq=config.max_len_seq,
            use_hetero_proj=config.use_hetero_proj,
        )

        # Deferred FastAPI app
        self.app: FastAPI | None = None

        # Future prediction modules (optional)
        self.use_future_prediction = bool(getattr(config, "use_future_prediction", False))
        self.predictor_hidden_dim = int(getattr(config, "predictor_hidden_dim", 1024))
        self.num_future_tokens = int(getattr(config, "num_future_tokens", 4))
        self.num_future_samples = int(getattr(config, "num_future_samples", 4))
        self.flow_sampling_steps = int(getattr(config, "flow_sampling_steps", 16))
        self.future_loss_weight = float(getattr(config, "future_loss_weight", 1.0))
        self.future_horizon_steps = int(getattr(config, "future_horizon_steps", 1))
        self.future_nll_weight = float(getattr(config, "future_nll_weight", 1.0))
        self.future_kl_weight = float(getattr(config, "future_kl_weight", 1.0))
        self.future_latent_dim = int(getattr(config, "future_latent_dim", 256))
        self.future_hidden_dim = int(getattr(config, "future_hidden_dim", 512))
        self.future_inject_layer_idx = int(getattr(config, "future_inject_layer_idx", -1))
        self.use_future_contrastive = bool(getattr(config, "use_future_contrastive", False))
        self.contrastive_loss_weight = float(getattr(config, "contrastive_loss_weight", 0.0))
        self.contrastive_num_layers = int(getattr(config, "contrastive_num_layers", 3))
        self.contrastive_proj_dim = int(getattr(config, "contrastive_proj_dim", 256))
        self.contrastive_temperature = float(getattr(config, "contrastive_temperature", 0.07))
        if self.use_future_prediction:
            self._init_future_modules(projection_dim)
        else:
            self.future_compressor = None
            self.future_predictor = None
            self.future_projector = None
            self.future_state_proj = None
            self.contrastive_projector = None

    def _init_future_modules(self, projection_dim: int) -> None:
        self.future_compressor = FutureTokenCompressor(
            hidden_dim=projection_dim,
            num_tokens=self.num_future_tokens,
        )
        self.future_predictor = RSSMFuturePredictor(
            cond_dim=self.num_future_tokens * projection_dim,
            latent_dim=self.future_latent_dim,
            hidden_dim=self.future_hidden_dim,
        )
        self.future_projector = FutureTokenProjector(hidden_dim=projection_dim)
        self.future_state_proj = torch.nn.Linear(self.dim_proprio, projection_dim)
        self.contrastive_projector = (
            ContrastiveProjector(hidden_dim=projection_dim, proj_dim=self.contrastive_proj_dim)
            if self.use_future_contrastive and self.contrastive_loss_weight > 0.0
            else None
        )

    def configure_future_prediction(
        self,
        enabled: bool,
        predictor_hidden_dim: int | None = None,
        num_future_tokens: int | None = None,
        num_future_samples: int | None = None,
        flow_sampling_steps: int | None = None,
        future_loss_weight: float | None = None,
        future_horizon_steps: int | None = None,
        future_nll_weight: float | None = None,
        future_kl_weight: float | None = None,
        future_latent_dim: int | None = None,
        future_hidden_dim: int | None = None,
        future_inject_layer_idx: int | None = None,
        use_future_contrastive: bool | None = None,
        contrastive_loss_weight: float | None = None,
        contrastive_num_layers: int | None = None,
        contrastive_proj_dim: int | None = None,
        contrastive_temperature: float | None = None,
    ) -> None:
        """Enable/disable future prediction modules post-load."""
        self.use_future_prediction = bool(enabled)
        self.config.use_future_prediction = bool(enabled)
        if predictor_hidden_dim is not None:
            self.predictor_hidden_dim = int(predictor_hidden_dim)
            self.config.predictor_hidden_dim = int(predictor_hidden_dim)
        if num_future_tokens is not None:
            self.num_future_tokens = int(num_future_tokens)
            self.config.num_future_tokens = int(num_future_tokens)
        if num_future_samples is not None:
            self.num_future_samples = int(num_future_samples)
            self.config.num_future_samples = int(num_future_samples)
        if flow_sampling_steps is not None:
            self.flow_sampling_steps = int(flow_sampling_steps)
            self.config.flow_sampling_steps = int(flow_sampling_steps)
        if future_loss_weight is not None:
            self.future_loss_weight = float(future_loss_weight)
            self.config.future_loss_weight = float(future_loss_weight)
        if future_horizon_steps is not None:
            self.future_horizon_steps = int(future_horizon_steps)
            self.config.future_horizon_steps = int(future_horizon_steps)
        if future_nll_weight is not None:
            self.future_nll_weight = float(future_nll_weight)
            self.config.future_nll_weight = float(future_nll_weight)
        if future_kl_weight is not None:
            self.future_kl_weight = float(future_kl_weight)
            self.config.future_kl_weight = float(future_kl_weight)
        if future_latent_dim is not None:
            self.future_latent_dim = int(future_latent_dim)
            self.config.future_latent_dim = int(future_latent_dim)
        if future_hidden_dim is not None:
            self.future_hidden_dim = int(future_hidden_dim)
            self.config.future_hidden_dim = int(future_hidden_dim)
        if future_inject_layer_idx is not None:
            self.future_inject_layer_idx = int(future_inject_layer_idx)
            self.config.future_inject_layer_idx = int(future_inject_layer_idx)
        if use_future_contrastive is not None:
            self.use_future_contrastive = bool(use_future_contrastive)
            self.config.use_future_contrastive = bool(use_future_contrastive)
        if contrastive_loss_weight is not None:
            self.contrastive_loss_weight = float(contrastive_loss_weight)
            self.config.contrastive_loss_weight = float(contrastive_loss_weight)
        if contrastive_num_layers is not None:
            self.contrastive_num_layers = int(contrastive_num_layers)
            self.config.contrastive_num_layers = int(contrastive_num_layers)
        if contrastive_proj_dim is not None:
            self.contrastive_proj_dim = int(contrastive_proj_dim)
            self.config.contrastive_proj_dim = int(contrastive_proj_dim)
        if contrastive_temperature is not None:
            self.contrastive_temperature = float(contrastive_temperature)
            self.config.contrastive_temperature = float(contrastive_temperature)
        if self.use_future_prediction:
            projection_dim = getattr(self.vlm.config, "projection_dim", None)
            if projection_dim is None:
                raise ValueError("Florence2 config must provide `projection_dim` for multimodal fusion.")
            self._init_future_modules(projection_dim)
        else:
            self.future_compressor = None
            self.future_predictor = None
            self.future_projector = None
            self.future_state_proj = None
            self.contrastive_projector = None

    def _resolve_future_inject_layer_idx(self) -> int:
        encoder_layers = len(self.vlm.language_model.model.encoder.layers)
        if encoder_layers <= 1:
            return 1
        idx = int(self.future_inject_layer_idx)
        if idx < 0:
            idx = max(1, encoder_layers // 3)
        return max(1, min(idx, encoder_layers - 1))

    def get_vlm_param_groups(self) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
        encoder = self.vlm.language_model.model.encoder
        inject_idx = self._resolve_future_inject_layer_idx()
        post_params = list(encoder.layers[inject_idx:].parameters()) if self.use_future_prediction else []
        post_ids = set(map(id, post_params))
        pre_params = [p for p in self.vlm.parameters() if id(p) not in post_ids]
        return pre_params, post_params

    def _prepare_vlm_inputs(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, V = pixel_values.shape[:2]
        flat_mask = image_mask.view(-1).to(torch.bool)
        flat_images = pixel_values.flatten(0, 1)

        num_valid = int(flat_mask.sum().item())
        if num_valid == 0:
            raise ValueError("At least one image view must be valid per batch.")

        valid_images = flat_images[flat_mask]
        valid_feats = self.vlm._encode_image(valid_images)
        N, D = valid_feats.shape[1:]

        image_features = valid_feats.new_zeros((B * V, N, D))
        image_features[flat_mask] = valid_feats
        image_features = image_features.view(B, V, N, D)

        inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
        merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
            image_features[:, 0],
            inputs_embeds,
        )
        aux_visual_inputs = image_features[:, 1:].reshape(B, -1, D)
        return merged_embeds, attention_mask, aux_visual_inputs

    def _expand_encoder_attention_mask(
        self,
        attention_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        encoder = self.vlm.language_model.model.encoder
        expanded_mask = attention_mask
        if expanded_mask is not None:
            if encoder._use_flash_attention_2:
                expanded_mask = expanded_mask if 0 in expanded_mask else None
            elif encoder._use_sdpa:
                expanded_mask = _prepare_4d_attention_mask_for_sdpa(expanded_mask, dtype)
            else:
                expanded_mask = _prepare_4d_attention_mask(expanded_mask, dtype)
        return expanded_mask

    def _prepare_encoder_hidden_states(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self.vlm.language_model.model.encoder
        input_shape = inputs_embeds[:, :, -1]
        embed_pos = encoder.embed_positions(input_shape).to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = encoder.layernorm_embedding(hidden_states)
        hidden_states = torch.nn.functional.dropout(hidden_states, p=encoder.dropout, training=encoder.training)
        return hidden_states, self._expand_encoder_attention_mask(attention_mask, inputs_embeds.dtype)

    def _run_encoder_layers(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        start_idx: int,
        end_idx: int,
        capture_indices: set[int] | None = None,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
        encoder = self.vlm.language_model.model.encoder
        captured: dict[int, torch.Tensor] = {}
        for idx in range(start_idx, end_idx):
            layer = encoder.layers[idx]
            if encoder.gradient_checkpointing and encoder.training:
                layer_outputs = encoder._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    None,
                    False,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    layer_head_mask=None,
                    output_attentions=False,
                )
            hidden_states = layer_outputs[0]
            if capture_indices and idx in capture_indices:
                captured[idx] = hidden_states
        return hidden_states, captured

    def _resolve_contrastive_layer_indices(self) -> list[int]:
        if not (self.use_future_contrastive and self.contrastive_loss_weight > 0.0):
            return []
        inject_idx = self._resolve_future_inject_layer_idx()
        if inject_idx <= 1:
            return []
        num = max(1, int(self.contrastive_num_layers))
        raw = np.linspace(0, inject_idx - 1, num=num, dtype=int).tolist()
        indices = sorted(set(int(x) for x in raw))
        return indices

    def _masked_pool(self, hidden_states: torch.Tensor, attention_mask_2d: torch.Tensor) -> torch.Tensor:
        mask = attention_mask_2d.to(hidden_states.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    def _symmetric_info_nce(self, current_embed: torch.Tensor, future_embed: torch.Tensor) -> torch.Tensor:
        if current_embed.size(0) <= 1:
            return current_embed.new_zeros(())
        current_embed = torch.nn.functional.normalize(current_embed, dim=-1)
        future_embed = torch.nn.functional.normalize(future_embed, dim=-1)
        logits = current_embed @ future_embed.t()
        logits = logits / max(self.contrastive_temperature, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            torch.nn.functional.cross_entropy(logits, labels) +
            torch.nn.functional.cross_entropy(logits.t(), labels)
        )

    def _compute_multidepth_contrastive_loss(
        self,
        current_captures: dict[int, torch.Tensor],
        current_attention_mask: torch.Tensor,
        future_captures: dict[int, torch.Tensor],
        future_attention_mask: torch.Tensor,
        batch_size: int,
        horizon: int,
    ) -> torch.Tensor | None:
        if self.contrastive_projector is None:
            return None
        layer_ids = sorted(set(current_captures.keys()) & set(future_captures.keys()))
        if not layer_ids:
            return None
        losses = []
        for layer_id in layer_ids:
            current_pooled = self._masked_pool(current_captures[layer_id], current_attention_mask)
            future_pooled = self._masked_pool(future_captures[layer_id], future_attention_mask)
            future_pooled = future_pooled.view(batch_size, horizon, -1).mean(dim=1)
            current_proj = self.contrastive_projector(current_pooled)
            future_proj = self.contrastive_projector(future_pooled.detach())
            losses.append(self._symmetric_info_nce(current_proj, future_proj))
        if not losses:
            return None
        return torch.stack(losses).mean()

    def _encode_to_injection_layer(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        image_mask: torch.Tensor,
        capture_indices: list[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
        merged_embeds, attention_mask_2d, aux_visual_inputs = self._prepare_vlm_inputs(input_ids, pixel_values, image_mask)
        hidden_states, attention_mask = self._prepare_encoder_hidden_states(merged_embeds, attention_mask_2d)
        inject_idx = self._resolve_future_inject_layer_idx()
        hidden_states, captures = self._run_encoder_layers(
            hidden_states,
            attention_mask,
            0,
            inject_idx,
            capture_indices=set(capture_indices or []),
        )
        return hidden_states, attention_mask_2d, aux_visual_inputs, captures

    def _project_future_tokens(
        self,
        current_hidden_states: torch.Tensor,
        current_attention_mask: torch.Tensor,
        input_ids: torch.LongTensor,
        proprio: torch.Tensor | None,
        current_contrastive_captures: dict[int, torch.Tensor] | None = None,
        future_image_input: torch.Tensor | None = None,
        future_image_mask: torch.Tensor | None = None,
        future_proprio: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        future_nll = None
        future_kl = None
        future_contrastive = None

        current_tokens = self.future_compressor(current_hidden_states)
        if self.future_state_proj is not None and proprio is not None:
            current_tokens = current_tokens + self.future_state_proj(proprio).unsqueeze(1)
        current_state = current_tokens.flatten(1)

        future_token_targets = None
        if future_image_input is not None and future_image_mask is not None:
            with torch.no_grad():
                if future_image_input.dim() == 5:
                    future_image_input = future_image_input.unsqueeze(1)
                if future_image_mask.dim() == 2:
                    future_image_mask = future_image_mask.unsqueeze(1)
                Bf, Hf, Vf, Cf, Hh, Ww = future_image_input.shape
                flat_images = future_image_input.reshape(Bf * Hf, Vf, Cf, Hh, Ww)
                flat_mask = future_image_mask.expand(Bf, Hf, Vf).reshape(Bf * Hf, Vf)
                input_ids_rep = input_ids[:, None, :].expand(Bf, Hf, input_ids.size(1)).reshape(Bf * Hf, input_ids.size(1))
                contrastive_indices = self._resolve_contrastive_layer_indices()
                future_hidden_states, future_attention_mask_2d, _, future_captures = self._encode_to_injection_layer(
                    input_ids_rep,
                    flat_images,
                    flat_mask,
                    capture_indices=contrastive_indices,
                )
                future_tokens = self.future_compressor(future_hidden_states).view(Bf, Hf, self.num_future_tokens, -1)
                if self.future_state_proj is not None and future_proprio is not None:
                    if future_proprio.dim() == 2:
                        future_proprio = future_proprio.unsqueeze(1).expand(Bf, Hf, future_proprio.size(-1))
                    future_tokens = future_tokens + self.future_state_proj(future_proprio).unsqueeze(2)
                future_token_targets = future_tokens.flatten(2)
            if current_contrastive_captures and future_captures and self.contrastive_projector is not None:
                future_contrastive = self._compute_multidepth_contrastive_loss(
                    current_captures=current_contrastive_captures,
                    current_attention_mask=current_attention_mask,
                    future_captures=future_captures,
                    future_attention_mask=future_attention_mask_2d,
                    batch_size=Bf,
                    horizon=Hf,
                )

        horizon = int(future_token_targets.size(1)) if future_token_targets is not None else int(self.future_horizon_steps)
        if future_token_targets is not None:
            recon, kl = self.future_predictor.elbo(current_state.float(), future_token_targets.float())
            future_nll = recon
            future_kl = kl

        sampled_rollouts, _, std = self.future_predictor.sample_rollout(
            current_state.float(),
            horizon=horizon,
            num_samples=max(1, int(self.num_future_samples)),
        )
        token_dim = current_hidden_states.size(-1)
        sampled_tokens = sampled_rollouts.view(
            sampled_rollouts.size(0),
            sampled_rollouts.size(1),
            sampled_rollouts.size(2),
            self.num_future_tokens,
            token_dim,
        ).mean(dim=2)
        std_tokens = std.view(std.size(0), std.size(1), self.num_future_tokens, token_dim).mean(dim=1)
        std_tokens = std_tokens.unsqueeze(1).expand(-1, sampled_tokens.size(1), -1, -1)
        future_tokens = self.future_projector(
            torch.cat([sampled_tokens, std_tokens], dim=-1)
        ).reshape(sampled_tokens.size(0), -1, token_dim)

        encoder = self.vlm.language_model.model.encoder
        pos_stub = torch.zeros(
            future_tokens.size(0),
            future_tokens.size(1),
            dtype=torch.long,
            device=future_tokens.device,
        )
        future_tokens = future_tokens + encoder.embed_positions(pos_stub).to(future_tokens.device)
        future_tokens = encoder.layernorm_embedding(future_tokens)
        future_tokens = torch.nn.functional.dropout(future_tokens, p=encoder.dropout, training=encoder.training)
        return future_tokens.to(current_hidden_states.dtype), future_nll, future_kl, future_contrastive

    # ============================= Florence2 encoder =============================
    def forward_vlm(
        self,
        input_ids: torch.LongTensor,        # [B, L]
        pixel_values: torch.FloatTensor,    # [B, V, C, H, W]
        image_mask: torch.Tensor,           # [B, V] (bool or 0/1)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode text + multi-view images via Florence2 encoder.

        Returns:
          { "vlm_features": [B, T_enc, D], "aux_visual_inputs": [B, (V-1)*N, D] }
        """
        hidden_states, attention_mask_2d, aux_visual_inputs, _ = self._encode_to_injection_layer(
            input_ids,
            pixel_values,
            image_mask,
        )

        if self.use_future_prediction and self.future_predictor is not None and self.future_compressor is not None:
            future_tokens, _, _, _ = self._project_future_tokens(
                current_hidden_states=hidden_states,
                current_attention_mask=attention_mask_2d,
                input_ids=input_ids,
                proprio=None,
            )
            hidden_states = torch.cat([hidden_states, future_tokens], dim=1)
            attention_mask_2d = torch.cat(
                [
                    attention_mask_2d,
                    attention_mask_2d.new_ones((attention_mask_2d.size(0), future_tokens.size(1))),
                ],
                dim=1,
            )

        attention_mask = self._expand_encoder_attention_mask(attention_mask_2d, hidden_states.dtype)
        hidden_states, _ = self._run_encoder_layers(
            hidden_states,
            attention_mask,
            self._resolve_future_inject_layer_idx(),
            len(self.vlm.language_model.model.encoder.layers),
        )
        return {"vlm_features": hidden_states, "aux_visual_inputs": aux_visual_inputs}

    # ================================= training =================================
    def forward(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        action: torch.Tensor,  # [B, T=num_actions, D=dim_action]
        future_image_input: torch.FloatTensor | None = None,
        future_image_mask: torch.Tensor | None = None,
        future_proprio: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        1) Encode multimodal inputs.
        2) Diffusion-style noisy mixture of actions: x_t = t*noise + (1-t)*gt.
        3) Space-specific preprocessing, prediction, and supervised loss.
        """
        contrastive_indices = self._resolve_contrastive_layer_indices()
        partial_hidden_states, attention_mask_2d, aux_visual_inputs, contrastive_captures = self._encode_to_injection_layer(
            input_ids,
            image_input,
            image_mask,
            capture_indices=contrastive_indices,
        )
        future_nll = None
        future_kl = None
        future_contrastive = None
        if self.use_future_prediction and self.future_predictor is not None and self.future_compressor is not None:
            future_tokens, future_nll, future_kl, future_contrastive = self._project_future_tokens(
                current_hidden_states=partial_hidden_states,
                current_attention_mask=attention_mask_2d,
                input_ids=input_ids,
                proprio=proprio,
                current_contrastive_captures=contrastive_captures,
                future_image_input=future_image_input,
                future_image_mask=future_image_mask,
                future_proprio=future_proprio,
            )
            partial_hidden_states = torch.cat([partial_hidden_states, future_tokens], dim=1)
            attention_mask_2d = torch.cat(
                [
                    attention_mask_2d,
                    attention_mask_2d.new_ones((attention_mask_2d.size(0), future_tokens.size(1))),
                ],
                dim=1,
            )

        attention_mask = self._expand_encoder_attention_mask(attention_mask_2d, partial_hidden_states.dtype)
        final_hidden_states, _ = self._run_encoder_layers(
            partial_hidden_states,
            attention_mask,
            self._resolve_future_inject_layer_idx(),
            len(self.vlm.language_model.model.encoder.layers),
        )
        enc = {
            "vlm_features": final_hidden_states,
            "aux_visual_inputs": aux_visual_inputs,
        }

        B = input_ids.shape[0]
        t = (torch.rand(1, device=input_ids.device)
             + torch.arange(B, device=input_ids.device) / B) % (1 - 1e-5)

        action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)

        pred_action = self.transformer(
            domain_id=domain_id,
            action_with_noise=action_noisy_m,
            t=t,
            proprio=proprio_m,
            **enc,
        )
        loss_dict = self.action_space.compute_loss(pred_action, action)
        if future_nll is not None:
            loss_dict["future_nll_loss"] = future_nll * self.future_nll_weight * self.future_loss_weight
        if future_kl is not None:
            loss_dict["future_kl_loss"] = future_kl * self.future_kl_weight * self.future_loss_weight
        if future_contrastive is not None:
            loss_dict["future_contrastive_loss"] = future_contrastive * self.contrastive_loss_weight
        return loss_dict

    # ================================= inference =================================
    @torch.no_grad()
    def generate_actions(
        self,
        input_ids: torch.LongTensor,
        image_input: torch.FloatTensor,
        image_mask: torch.Tensor,
        domain_id: torch.LongTensor,
        proprio: torch.Tensor,
        steps: int = 10,
    ) -> torch.Tensor:
        """
        Iterative denoising (linear schedule).
        Applies action_space.postprocess at the end (e.g., sigmoid on gripper).
        """
        self.eval()
        partial_hidden_states, attention_mask_2d, aux_visual_inputs, _ = self._encode_to_injection_layer(
            input_ids,
            image_input,
            image_mask,
        )
        if self.use_future_prediction and self.future_predictor is not None and self.future_compressor is not None:
            future_tokens, _, _, _ = self._project_future_tokens(
                current_hidden_states=partial_hidden_states,
                current_attention_mask=attention_mask_2d,
                input_ids=input_ids,
                proprio=proprio,
            )
            partial_hidden_states = torch.cat([partial_hidden_states, future_tokens], dim=1)
            attention_mask_2d = torch.cat(
                [
                    attention_mask_2d,
                    attention_mask_2d.new_ones((attention_mask_2d.size(0), future_tokens.size(1))),
                ],
                dim=1,
            )

        attention_mask = self._expand_encoder_attention_mask(attention_mask_2d, partial_hidden_states.dtype)
        final_hidden_states, _ = self._run_encoder_layers(
            partial_hidden_states,
            attention_mask,
            self._resolve_future_inject_layer_idx(),
            len(self.vlm.language_model.model.encoder.layers),
        )
        enc = {
            "vlm_features": final_hidden_states,
            "aux_visual_inputs": aux_visual_inputs,
        }

        B = input_ids.shape[0]
        D = self.action_space.dim_action

        x1 = torch.randn(B, self.num_actions, D, device=proprio.device, dtype=proprio.dtype)
        action = torch.zeros_like(x1)

        steps = max(1, int(steps))
        for i in range(steps, 0, -1):
            t = torch.full((B,), i / steps, device=proprio.device, dtype=proprio.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
            proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
            action = self.transformer(
                domain_id=domain_id,
                action_with_noise=x_t_m,
                proprio=proprio_m,
                t=t,
                **enc,
            )
        return self.action_space.postprocess(action)

    # =============================== FastAPI service =============================
    def _normalize_inference_proprio(self, proprio: np.ndarray) -> np.ndarray:
        stats = self.inference_normalization
        if not stats:
            return proprio.astype(np.float32)
        if stats.get("mode") != "mean_std":
            raise ValueError(f"Unsupported inference normalization mode: {stats.get('mode')}")
        real_dim = int(getattr(self.action_space, "real_dim", len(stats["mean"])))
        mean = np.asarray(stats["mean"], dtype=np.float32)[:real_dim]
        std = np.maximum(np.asarray(stats["std"], dtype=np.float32)[:real_dim], 1e-6)
        x = proprio.astype(np.float32).copy()
        x[..., :real_dim] = (x[..., :real_dim] - mean) / std
        return x

    def _unnormalize_inference_action(self, action: np.ndarray) -> np.ndarray:
        stats = self.inference_normalization
        if not stats:
            return action.astype(np.float32)
        if stats.get("mode") != "mean_std":
            raise ValueError(f"Unsupported inference normalization mode: {stats.get('mode')}")
        real_dim = int(getattr(self.action_space, "real_dim", len(stats["mean"])))
        mean = np.asarray(stats["mean"], dtype=np.float32)[:real_dim]
        std = np.maximum(np.asarray(stats["std"], dtype=np.float32)[:real_dim], 1e-6)
        x = action.astype(np.float32).copy()
        x[..., :real_dim] = x[..., :real_dim] * std + mean
        return x

    def _build_app(self, processor):
        """
        Minimal FastAPI app for XVLA inference.

        Args:
            processor: callable(images, text) -> Dict[str, torch.Tensor]
                       expected keys: "input_ids", "image_input", "image_mask"
        """
        if self.app is not None:
            return

        app = FastAPI()

        @app.post("/act")
        def act(payload: Dict[str, Any]):
            try:
                self.eval()
                # Decode up to 3 image inputs
                images = []
                for key in ("image0", "image1", "image2"):
                    if key not in payload: continue
                    v = json_numpy.loads(payload[key])
                    if isinstance(v, np.ndarray):
                        if v.ndim == 1:  # encoded bytes
                            v = cv2.imdecode(v, cv2.IMREAD_COLOR)
                        images.append(Image.fromarray(v))
                    elif isinstance(v, (list, tuple)):
                        images.append(Image.fromarray(np.array(v)))
                    elif isinstance(v, str):
                        images.append(Image.open(v))
                if not images:
                    return JSONResponse({"error": "No valid images found."}, status_code=400)

                # Multimodal preprocessing by processor
                inputs = processor(images, payload["language_instruction"])
                if not {"input_ids", "image_input", "image_mask"}.issubset(inputs):
                    return JSONResponse({"error": "Processor returned incomplete inputs."}, status_code=400)

                # Build proprio/domain tensors
                proprio_np = np.asarray(json_numpy.loads(payload["proprio"]), dtype=np.float32)
                proprio_np = self._normalize_inference_proprio(proprio_np)
                proprio = torch.as_tensor(proprio_np)
                domain_id = torch.tensor([int(payload["domain_id"])], dtype=torch.long)

                # Align to model's device/dtype
                device = next(self.parameters()).device
                dtype = next(self.parameters()).dtype

                def to_model(t: torch.Tensor) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        t = torch.as_tensor(t)
                    # cast floats to model dtype, keep integral/bool as-is
                    return t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)

                inputs = {k: to_model(v) for k, v in inputs.items()}
                inputs.update({
                    "proprio": to_model(proprio.unsqueeze(0)),
                    "domain_id": domain_id.to(device),
                })

                # Inference
                steps = int(payload.get("steps", 10))
                action = self.generate_actions(**inputs, steps=steps).squeeze(0).float().cpu().numpy()
                action = self._unnormalize_inference_action(action)
                return JSONResponse({"action": action.tolist()})

            except Exception:
                logging.error(traceback.format_exc())
                return JSONResponse({"error": "Request failed"}, status_code=400)

        self.app = app

    def run(self, processor, host: str = "0.0.0.0", port: int = 8000):
        """
        Launch the FastAPI service.
        """
        self._build_app(processor)
        assert self.app is not None
        uvicorn.run(self.app, host=host, port=port)
