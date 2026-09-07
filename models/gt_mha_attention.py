"""GT-MHA implementation vendored from shahils01/Attention-is-Transformed.

Source branch: vision-deit
The implementation is kept intact so DiWINE-VLA checkpoints remain
self-contained on Palmetto.
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings(
    "ignore",
    message=r".*An output with one or more elements was resized.*",
    category=UserWarning,
)


def _negative_large(dtype: torch.dtype) -> float:
    if dtype in (torch.float16, torch.bfloat16):
        return -1e4
    return -1e9


KVCache = tuple[torch.Tensor, torch.Tensor]


def _append_base_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    past_key_value: KVCache | None,
) -> KVCache:
    """Append base K/V tensors stored as [batch, bases, time, dim]."""
    if past_key_value is None:
        return key, value
    past_key, past_value = past_key_value
    if past_key.ndim != 4 or past_value.ndim != 4:
        raise ValueError("cached keys and values must have shape [batch, bases, time, dim]")
    if past_key.shape[:2] + past_key.shape[3:] != key.shape[:2] + key.shape[3:]:
        raise ValueError("cached key shape is incompatible with the current base projection")
    if past_value.shape[:2] + past_value.shape[3:] != value.shape[:2] + value.shape[3:]:
        raise ValueError("cached value shape is incompatible with the current base projection")
    return torch.cat((past_key, key), dim=2), torch.cat((past_value, value), dim=2)


class LieGeneratedMetricAttention(nn.Module):
    """Multi-head attention with shared Q/K/V and Lie-generated head metrics."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        head_dim: int,
        num_generators: int,
        dropout: float = 0.0,
        bias: bool = False,
        generator_type: str = "full",
        generator_mixing: str = "softmax",
        use_sdpa: bool = True,
        causal: bool = False,
        stabilize_generators: bool = True,
        normalize_generators: bool = False,
        head_generator_symmetric_cap: float | None = None,
        theta_init_scale: float = 4.0,
        generator_init_scale: float = 0.02,
        base_dim: int | None = None,
        value_dim: int | None = None,
        metric_mode: str = "exp",
        metric_beta: float = 1.0,
        metric_clip_min: float | None = None,
        metric_clip_max: float | None = None,
        value_beta: float | None = None,
        theta_init: str = "balanced_simplex",
        logit_scale_mode: str = "sqrt_dim",
        learn_head_temperature: bool = False,
        value_transform: str = "none",
        num_base_heads: int = 1,
        num_value_base_heads: int | None = None,
        fuse_base_qkv: bool = False,
        fold_value_transform_into_output: bool = False,
        sdpa_gqa_mode: str = "auto",
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or head_dim <= 0 or num_generators <= 0:
            raise ValueError("d_model, num_heads, head_dim, and num_generators must be positive")
        if generator_type not in {"full", "diagonal", "symmetric"}:
            raise ValueError(f"unsupported generator_type: {generator_type}")
        if generator_mixing not in {"softmax", "none"}:
            raise ValueError(f"unsupported generator_mixing: {generator_mixing}")
        if metric_mode not in {"exp", "residual", "quadratic", "unconstrained"}:
            raise ValueError(f"unsupported metric_mode: {metric_mode}")
        if theta_init not in {"balanced_simplex", "random_sphere", "circle"}:
            raise ValueError(f"unsupported theta_init: {theta_init}")
        if logit_scale_mode not in {"sqrt_dim", "rms_metric"}:
            raise ValueError(f"unsupported logit_scale_mode: {logit_scale_mode}")
        if value_transform not in {
            "none",
            "diag",
            "lie",
            "lie_exp",
            "lie_residual",
            "lie_quadratic",
            "unconstrained",
        }:
            raise ValueError(f"unsupported value_transform: {value_transform}")
        if metric_mode == "unconstrained" and generator_type == "diagonal":
            raise ValueError("unconstrained metric_mode requires dense metrics")
        if metric_mode == "unconstrained" and head_generator_symmetric_cap is not None:
            raise ValueError(
                "head_generator_symmetric_cap is unavailable for unconstrained metric_mode"
            )
        if metric_clip_min is not None and metric_clip_min < 0:
            raise ValueError("metric_clip_min must be non-negative")
        if head_generator_symmetric_cap is not None and head_generator_symmetric_cap <= 0:
            raise ValueError("head_generator_symmetric_cap must be positive")
        if (
            metric_clip_min is not None
            and metric_clip_max is not None
            and metric_clip_min > metric_clip_max
        ):
            raise ValueError("metric_clip_min must be <= metric_clip_max")
        if num_base_heads <= 0:
            raise ValueError("num_base_heads must be positive")
        if num_heads % num_base_heads != 0:
            raise ValueError("num_heads must be divisible by num_base_heads")
        if num_value_base_heads is None:
            num_value_base_heads = num_base_heads
        if num_value_base_heads <= 0:
            raise ValueError("num_value_base_heads must be positive")
        if num_heads % num_value_base_heads != 0:
            raise ValueError("num_heads must be divisible by num_value_base_heads")
        if sdpa_gqa_mode not in {"auto", "native", "expand"}:
            raise ValueError("sdpa_gqa_mode must be one of: auto, native, expand")

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_base_heads = num_base_heads
        self.generated_heads_per_base = num_heads // num_base_heads
        self.num_value_base_heads = num_value_base_heads
        self.generated_value_heads_per_base = num_heads // num_value_base_heads
        self.head_dim = head_dim
        self.base_dim = base_dim if base_dim is not None else head_dim
        self.value_dim = value_dim if value_dim is not None else head_dim
        if self.base_dim <= 0 or self.value_dim <= 0:
            raise ValueError("base_dim and value_dim must be positive")
        self.num_generators = num_generators
        self.dropout = dropout
        self.generator_type = generator_type
        self.generator_mixing = generator_mixing
        self.use_sdpa = use_sdpa
        self.causal = causal
        self.stabilize_generators = stabilize_generators
        self.normalize_generators = normalize_generators
        self.head_generator_symmetric_cap = head_generator_symmetric_cap
        self.theta_init_scale = theta_init_scale
        self.generator_init_scale = generator_init_scale
        self.metric_mode = metric_mode
        self.metric_beta = metric_beta
        self.metric_clip_min = metric_clip_min
        self.metric_clip_max = metric_clip_max
        self.value_beta = metric_beta if value_beta is None else value_beta
        self.theta_init = theta_init
        self.logit_scale_mode = logit_scale_mode
        self.learn_head_temperature = learn_head_temperature
        self.value_transform = value_transform
        self.value_transform_mode = self._resolve_value_transform_mode(value_transform, metric_mode)
        self.fuse_base_qkv = fuse_base_qkv
        self.fold_value_transform_into_output = fold_value_transform_into_output
        self.sdpa_gqa_mode = sdpa_gqa_mode

        self.q_proj = nn.Linear(d_model, self.num_base_heads * self.base_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, self.num_base_heads * self.base_dim, bias=bias)
        self.v_proj = nn.Linear(
            d_model, self.num_value_base_heads * self.value_dim, bias=bias
        )
        self.out_proj = nn.Linear(num_heads * self.value_dim, d_model, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

        if metric_mode == "unconstrained":
            self.raw_metrics = nn.Parameter(torch.empty(num_heads, self.base_dim, self.base_dim))
        elif generator_type == "diagonal":
            self.generators = nn.Parameter(torch.empty(num_generators, self.base_dim))
        else:
            self.generators = nn.Parameter(torch.empty(num_generators, self.base_dim, self.base_dim))
        if metric_mode != "unconstrained":
            self.theta = nn.Parameter(torch.empty(num_heads, num_generators))
        if learn_head_temperature:
            self.head_logit_scale = nn.Parameter(torch.ones(num_heads))
        if self.value_transform_mode == "diag":
            self.value_scale = nn.Parameter(torch.ones(num_heads, self.value_dim))
        elif self.value_transform_mode == "unconstrained":
            self.raw_value_transforms = nn.Parameter(
                torch.empty(num_heads, self.value_dim, self.value_dim)
            )
        elif self.value_transform_mode in {"exp", "residual", "quadratic"}:
            if generator_type == "diagonal":
                self.value_generators = nn.Parameter(torch.empty(num_generators, self.value_dim))
            else:
                self.value_generators = nn.Parameter(
                    torch.empty(num_generators, self.value_dim, self.value_dim)
                )
            self.value_theta = nn.Parameter(torch.empty(num_heads, num_generators))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)
            nn.init.zeros_(self.v_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

        if self.metric_mode == "unconstrained":
            nn.init.zeros_(self.raw_metrics)
        else:
            generator_std = self.generator_init_scale / math.sqrt(self.base_dim)
            nn.init.normal_(self.generators, mean=0.0, std=generator_std)
            self._init_theta()
        if hasattr(self, "head_logit_scale"):
            nn.init.ones_(self.head_logit_scale)
        if hasattr(self, "value_scale"):
            nn.init.ones_(self.value_scale)
        if hasattr(self, "raw_value_transforms"):
            nn.init.zeros_(self.raw_value_transforms)
        if hasattr(self, "value_generators"):
            generator_std = self.generator_init_scale / math.sqrt(self.value_dim)
            nn.init.normal_(self.value_generators, mean=0.0, std=generator_std)
            self._init_head_coordinates(self.value_theta)

    @staticmethod
    def _resolve_value_transform_mode(value_transform: str, metric_mode: str) -> str:
        if value_transform == "lie":
            if metric_mode == "unconstrained":
                return "unconstrained"
            return metric_mode
        if value_transform.startswith("lie_"):
            return value_transform.removeprefix("lie_")
        return value_transform

    def _init_theta(self) -> None:
        self._init_head_coordinates(self.theta)

    def _init_head_coordinates(self, coordinates: torch.Tensor) -> None:
        if self.theta_init_scale <= 0:
            nn.init.zeros_(coordinates)
            return

        with torch.no_grad():
            if self.theta_init == "balanced_simplex":
                if self.num_generators == 1:
                    # There is no head-diversity degree of freedom with one
                    # generator. Keep the sole coefficient nonzero for raw
                    # mixing; softmax mixing maps it to one regardless.
                    directions = torch.ones_like(coordinates)
                else:
                    # Vertices of the centered regular simplex maximize the
                    # pairwise distance between generator-mixture logits.
                    # Cycling through vertices keeps their occupancy balanced
                    # and guarantees distinct heads within each base whenever
                    # generated_heads_per_base <= num_generators.
                    assignments = torch.arange(
                        self.num_heads,
                        device=coordinates.device,
                    ).remainder(self.num_generators)
                    directions = torch.full_like(
                        coordinates,
                        -1.0 / self.num_generators,
                    )
                    directions.scatter_(
                        1,
                        assignments[:, None],
                        1.0 - 1.0 / self.num_generators,
                    )
            elif self.theta_init == "circle" and self.num_generators >= 2:
                head_ids = torch.arange(
                    self.num_heads,
                    device=coordinates.device,
                    dtype=coordinates.dtype,
                )
                base_ids = torch.div(
                    head_ids,
                    self.generated_heads_per_base,
                    rounding_mode="floor",
                )
                generated_ids = head_ids.remainder(self.generated_heads_per_base)
                angles = (
                    2 * math.pi * generated_ids / self.generated_heads_per_base
                    + math.pi * base_ids / max(self.generated_heads_per_base, 1)
                )
                directions = torch.zeros_like(coordinates)
                directions[:, 0] = torch.cos(angles)
                directions[:, 1] = torch.sin(angles)
                if self.num_generators > 2:
                    noise = torch.randn(
                        self.num_heads,
                        self.num_generators - 2,
                        device=coordinates.device,
                        dtype=coordinates.dtype,
                    )
                    directions[:, 2:] = 0.01 * noise
            else:
                directions = torch.randn_like(coordinates)
            directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            coordinates.copy_(directions * self.theta_init_scale)

    def _dense_generators(self) -> torch.Tensor:
        if self.generator_type == "diagonal":
            return torch.diag_embed(self._maybe_normalize_generators(self.generators))

        generators = self.generators
        if self.generator_type == "symmetric":
            generators = 0.5 * (generators + generators.transpose(-1, -2))
        if self.stabilize_generators:
            generators = self._make_trace_zero(generators)
        return self._maybe_normalize_generators(generators)

    def _dense_value_generators(self) -> torch.Tensor:
        if self.generator_type == "diagonal":
            return torch.diag_embed(self._maybe_normalize_generators(self.value_generators))

        generators = self.value_generators
        if self.generator_type == "symmetric":
            generators = 0.5 * (generators + generators.transpose(-1, -2))
        if self.stabilize_generators:
            generators = self._make_trace_zero(generators)
        return self._maybe_normalize_generators(generators)

    def _maybe_normalize_generators(self, generators: torch.Tensor) -> torch.Tensor:
        if not self.normalize_generators:
            return generators
        return self._normalize_generators(generators)

    @staticmethod
    def _normalize_generators(generators: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norms = generators.float().reshape(generators.shape[0], -1).norm(dim=-1)
        scale = norms.clamp_min(eps).to(dtype=generators.dtype)
        return generators / scale.view(-1, *([1] * (generators.ndim - 1)))

    @staticmethod
    def _make_trace_zero(generators: torch.Tensor) -> torch.Tensor:
        dim = generators.shape[-1]
        trace = generators.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        eye = torch.eye(dim, device=generators.device, dtype=generators.dtype)
        return generators - (trace[..., None, None] / dim) * eye

    def compute_head_generators(self) -> torch.Tensor:
        """Return dense per-head A_h generators after applying metric_beta."""
        if self.metric_mode == "unconstrained":
            return self.metric_beta * self.raw_metrics
        if self.generator_type == "diagonal":
            return torch.diag_embed(self._compute_metric_diagonal())
        generators = self._dense_generators()
        head_generators = torch.einsum("hm,mde->hde", self.metric_theta_weights(), generators)
        head_generators = self.metric_beta * head_generators
        if self.generator_type == "symmetric":
            head_generators = 0.5 * (head_generators + head_generators.transpose(-1, -2))
        return self._maybe_cap_head_generator_symmetric_part(head_generators)

    def _compute_metric_diagonal(self) -> torch.Tensor:
        diagonal = torch.einsum(
            "hm,md->hd",
            self.metric_theta_weights(),
            self._maybe_normalize_generators(self.generators),
        )
        diagonal = self.metric_beta * diagonal
        if self.head_generator_symmetric_cap is None:
            return diagonal
        norms = diagonal.float().norm(dim=-1, keepdim=True)
        scale = (self.head_generator_symmetric_cap / norms.clamp_min(1e-8)).clamp(max=1.0)
        return diagonal * scale.to(dtype=diagonal.dtype)

    def _maybe_cap_head_generator_symmetric_part(
        self,
        head_generators: torch.Tensor,
    ) -> torch.Tensor:
        """Radially cap sym(A_h), leaving the skew component unchanged."""
        if self.head_generator_symmetric_cap is None:
            return head_generators
        symmetric = 0.5 * (head_generators + head_generators.transpose(-1, -2))
        skew = 0.5 * (head_generators - head_generators.transpose(-1, -2))
        norms = symmetric.float().norm(dim=(-2, -1), keepdim=True)
        scale = (self.head_generator_symmetric_cap / norms.clamp_min(1e-8)).clamp(max=1.0)
        return skew + symmetric * scale.to(dtype=head_generators.dtype)

    def pre_cap_head_generator_symmetric_norms(self) -> torch.Tensor | None:
        """Return per-head symmetric norms before applying the configured cap."""
        if self.head_generator_symmetric_cap is None or self.metric_mode == "unconstrained":
            return None
        if self.generator_type == "diagonal":
            diagonal = torch.einsum(
                "hm,md->hd",
                self.metric_theta_weights(),
                self._maybe_normalize_generators(self.generators),
            )
            return (self.metric_beta * diagonal).float().norm(dim=-1)

        generators = self._dense_generators()
        head_generators = torch.einsum("hm,mde->hde", self.metric_theta_weights(), generators)
        head_generators = self.metric_beta * head_generators
        if self.generator_type == "symmetric":
            head_generators = 0.5 * (
                head_generators + head_generators.transpose(-1, -2)
            )
        symmetric = 0.5 * (head_generators + head_generators.transpose(-1, -2))
        return symmetric.float().norm(dim=(-2, -1))

    def metric_theta_weights(self) -> torch.Tensor:
        """Return configured generator coefficients for each metric head."""
        if not hasattr(self, "theta"):
            raise AttributeError("unconstrained metric mode does not use theta")
        return self._generator_mixing_weights(self.theta)

    def value_theta_weights(self) -> torch.Tensor:
        """Return configured generator coefficients for each value head."""
        if not hasattr(self, "value_theta"):
            raise AttributeError("value transform mode does not use value_theta")
        return self._generator_mixing_weights(self.value_theta)

    def _generator_mixing_weights(self, coordinates: torch.Tensor) -> torch.Tensor:
        if self.generator_mixing == "softmax":
            return torch.softmax(coordinates, dim=-1)
        return coordinates

    def _clip_metric_singular_values(self, metrics: torch.Tensor) -> torch.Tensor:
        if self.metric_clip_min is None and self.metric_clip_max is None:
            return metrics
        u, singular_values, vh = torch.linalg.svd(metrics.float(), full_matrices=False)
        clipped = singular_values
        if self.metric_clip_min is not None:
            clipped = clipped.clamp_min(self.metric_clip_min)
        if self.metric_clip_max is not None:
            clipped = clipped.clamp_max(self.metric_clip_max)
        clipped_metrics = u @ torch.diag_embed(clipped) @ vh
        return clipped_metrics.to(dtype=metrics.dtype)

    def compute_metrics(self) -> torch.Tensor:
        if self.metric_mode == "unconstrained":
            eye = torch.eye(self.base_dim, device=self.raw_metrics.device, dtype=self.raw_metrics.dtype)
            metrics = eye[None, :, :] + self.metric_beta * self.raw_metrics
            return self._clip_metric_singular_values(metrics)

        if self.generator_type == "diagonal":
            diagonal = self._compute_metric_diagonal()
            if self.metric_mode == "residual":
                metrics = torch.diag_embed(1.0 + diagonal)
                return self._clip_metric_singular_values(metrics)
            if self.metric_mode == "quadratic":
                metrics = torch.diag_embed(1.0 + diagonal + 0.5 * diagonal.square())
                return self._clip_metric_singular_values(metrics)
            scale = torch.exp(diagonal)
            metrics = torch.diag_embed(scale)
            return self._clip_metric_singular_values(metrics)

        head_generators = self.compute_head_generators()
        if self.metric_mode == "residual":
            eye = torch.eye(
                self.base_dim,
                device=head_generators.device,
                dtype=head_generators.dtype,
            )
            metrics = eye[None, :, :] + head_generators
            return self._clip_metric_singular_values(metrics)
        if self.metric_mode == "quadratic":
            eye = torch.eye(
                self.base_dim,
                device=head_generators.device,
                dtype=head_generators.dtype,
            )
            second_order = torch.matmul(head_generators, head_generators)
            metrics = eye[None, :, :] + head_generators + 0.5 * second_order
            return self._clip_metric_singular_values(metrics)
        metrics = torch.linalg.matrix_exp(head_generators.float()).to(dtype=head_generators.dtype)
        return self._clip_metric_singular_values(metrics)

    def effective_generators(self) -> torch.Tensor:
        """Return dense stabilized generator basis for diagnostics."""
        if self.metric_mode == "unconstrained":
            return self.raw_metrics
        return self._dense_generators()

    def compute_value_transforms(self) -> torch.Tensor:
        if self.value_transform_mode == "none":
            eye = torch.eye(
                self.value_dim,
                device=self.v_proj.weight.device,
                dtype=self.v_proj.weight.dtype,
            )
            return eye[None, :, :].expand(self.num_heads, self.value_dim, self.value_dim)
        if self.value_transform_mode == "diag":
            return torch.diag_embed(self.value_scale)
        if self.value_transform_mode == "unconstrained":
            eye = torch.eye(
                self.value_dim,
                device=self.raw_value_transforms.device,
                dtype=self.raw_value_transforms.dtype,
            )
            return eye[None, :, :] + self.value_beta * self.raw_value_transforms

        if self.generator_type == "diagonal":
            diagonal = torch.einsum(
                "hm,md->hd",
                self.value_theta_weights(),
                self._maybe_normalize_generators(self.value_generators),
            )
            diagonal = self.value_beta * diagonal
            if self.value_transform_mode == "residual":
                return torch.diag_embed(1.0 + diagonal)
            if self.value_transform_mode == "quadratic":
                return torch.diag_embed(1.0 + diagonal + 0.5 * diagonal.square())
            scale = torch.exp(diagonal)
            return torch.diag_embed(scale)

        generators = self._dense_value_generators()
        head_generators = torch.einsum("hm,mde->hde", self.value_theta_weights(), generators)
        head_generators = self.value_beta * head_generators
        if self.generator_type == "symmetric":
            head_generators = 0.5 * (head_generators + head_generators.transpose(-1, -2))
        if self.value_transform_mode == "residual":
            eye = torch.eye(
                self.value_dim,
                device=head_generators.device,
                dtype=head_generators.dtype,
            )
            return eye[None, :, :] + head_generators
        if self.value_transform_mode == "quadratic":
            eye = torch.eye(
                self.value_dim,
                device=head_generators.device,
                dtype=head_generators.dtype,
            )
            second_order = torch.matmul(head_generators, head_generators)
            return eye[None, :, :] + head_generators + 0.5 * second_order
        return torch.linalg.matrix_exp(head_generators.float()).to(dtype=head_generators.dtype)

    def _project(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # A fused projection is only valid for self-attention. Cross-attention
        # has decoder queries and encoder keys/values from different tensors.
        if self.fuse_base_qkv and context is None:
            # Retain the legacy Parameters and state-dict keys so existing model
            # and optimizer checkpoints remain directly loadable.
            weight = torch.cat(
                (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight), dim=0
            )
            bias = None
            if self.q_proj.bias is not None:
                bias = torch.cat(
                    (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias), dim=0
                )
            qkv = F.linear(x, weight, bias)
            qk_width = self.num_base_heads * self.base_dim
            value_width = self.num_value_base_heads * self.value_dim
            return torch.split(qkv, (qk_width, qk_width, value_width), dim=-1)
        key_value = x if context is None else context
        return self.q_proj(x), self.k_proj(key_value), self.v_proj(key_value)

    def _reshape_base_qk(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(batch, seq_len, self.num_base_heads, self.base_dim).transpose(1, 2)

    def _reshape_base_values(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = tensor.shape
        return tensor.view(
            batch, seq_len, self.num_value_base_heads, self.value_dim
        ).transpose(1, 2)

    def _metrics_by_base(self, metrics: torch.Tensor | None = None) -> torch.Tensor:
        if metrics is None:
            metrics = self.compute_metrics()
        return metrics.view(
            self.num_base_heads,
            self.generated_heads_per_base,
            self.base_dim,
            self.base_dim,
        )

    def _apply_metric_to_queries(
        self,
        q: torch.Tensor,
        metrics: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q_base = self._reshape_base_qk(q)
        if (
            metrics is None
            and self.generator_type == "diagonal"
            and self.metric_mode != "unconstrained"
            and self.metric_clip_min is None
            and self.metric_clip_max is None
        ):
            diagonal = self._compute_metric_diagonal()
            if self.metric_mode == "residual":
                scale = 1.0 + diagonal
            elif self.metric_mode == "quadratic":
                scale = 1.0 + diagonal + 0.5 * diagonal.square()
            else:
                scale = torch.exp(diagonal)
            scale = scale.view(
                self.num_base_heads,
                self.generated_heads_per_base,
                self.base_dim,
            )
            q_metric = q_base[:, :, None, :, :] * scale[None, :, :, None, :]
        else:
            metrics_by_base = self._metrics_by_base(metrics)
            q_metric = torch.einsum("nbtd,bkde->nbkte", q_base, metrics_by_base)
        batch, _, _, seq_len, _ = q_metric.shape
        return q_metric.reshape(batch, self.num_heads, seq_len, self.base_dim)

    def _score_scale(self, metrics: torch.Tensor | None = None) -> torch.Tensor | float:
        if self.logit_scale_mode == "sqrt_dim" and not hasattr(self, "head_logit_scale"):
            return math.sqrt(self.base_dim)

        device = self.q_proj.weight.device
        dtype = self.q_proj.weight.dtype
        denom = torch.full((self.num_heads,), math.sqrt(self.base_dim), device=device, dtype=dtype)
        if self.logit_scale_mode == "rms_metric":
            if metrics is None:
                metrics = self.compute_metrics()
            rms = metrics.float().pow(2).sum(dim=(-1, -2)).div(self.base_dim).sqrt()
            denom = denom * rms.to(device=device, dtype=dtype).clamp_min(1e-6)
        if hasattr(self, "head_logit_scale"):
            denom = denom / self.head_logit_scale.to(device=device, dtype=dtype).clamp_min(1e-6)
        return denom

    def compute_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        apply_scale: bool = True,
    ) -> torch.Tensor:
        metrics = (
            self.compute_metrics()
            if apply_scale and self.logit_scale_mode == "rms_metric"
            else None
        )
        q_metric = self._apply_metric_to_queries(q, metrics=metrics)
        k_heads = self._expand_keys(k)
        scores = torch.einsum("bhtd,bhsd->bhts", q_metric, k_heads)
        if not apply_scale:
            return scores
        scale = self._score_scale(metrics=metrics)
        if isinstance(scale, torch.Tensor):
            return scores / scale[None, :, None, None].to(device=scores.device, dtype=scores.dtype)
        return scores / scale

    def _expand_values(self, v: torch.Tensor) -> torch.Tensor:
        if v.ndim == 3:
            values = self._reshape_base_values(v)
        elif v.ndim == 4:
            values = v
        else:
            raise ValueError("values must have shape [batch, time, bases*dim] or [batch, bases, time, dim]")
        batch, _, seq_len, _ = values.shape
        values = values[:, :, None, :, :].expand(
            batch,
            self.num_value_base_heads,
            self.generated_value_heads_per_base,
            seq_len,
            self.value_dim,
        )
        return values.reshape(batch, self.num_heads, seq_len, self.value_dim)

    def _apply_value_transforms_to_outputs(self, values: torch.Tensor) -> torch.Tensor:
        """Apply linear head transforms after aggregation instead of to every cached token."""
        batch, _, seq_len, _ = values.shape
        values = values.reshape(
            batch,
            self.num_value_base_heads,
            self.generated_value_heads_per_base,
            seq_len,
            self.value_dim,
        )
        if self.value_transform_mode in {"exp", "residual", "quadratic"} and self.generator_type == "diagonal":
            diagonal = torch.einsum(
                "hm,md->hd",
                self.value_theta_weights(),
                self._maybe_normalize_generators(self.value_generators),
            )
            diagonal = self.value_beta * diagonal
            if self.value_transform_mode == "residual":
                scale = 1.0 + diagonal
            elif self.value_transform_mode == "quadratic":
                scale = 1.0 + diagonal + 0.5 * diagonal.square()
            else:
                scale = torch.exp(diagonal)
            scale = scale.view(
                self.num_value_base_heads,
                self.generated_value_heads_per_base,
                self.value_dim,
            )
            values = values * scale[None, :, :, None, :]
        elif self.value_transform_mode in {"exp", "residual", "quadratic", "unconstrained"}:
            transforms = self.compute_value_transforms().view(
                self.num_value_base_heads,
                self.generated_value_heads_per_base,
                self.value_dim,
                self.value_dim,
            )
            values = torch.einsum("nbktd,bkde->nbkte", values, transforms)
        values = values.reshape(batch, self.num_heads, seq_len, self.value_dim)
        if self.value_transform_mode == "diag":
            values = values * self.value_scale[None, :, None, :]
        return values

    def _output_projection(self, out_heads: torch.Tensor) -> torch.Tensor:
        """Apply the value transform and output projection with optional folding.

        For row-vector head outputs ``Z_h`` and transforms ``R_h``,
        ``(Z_h R_h) W_{O,h} = Z_h (R_h W_{O,h})``. Folding is therefore
        algebraically exact and preserves all existing Parameter tensors and
        state-dict keys.
        """
        batch, _, seq_len, _ = out_heads.shape
        if (
            not self.fold_value_transform_into_output
            or self.value_transform_mode == "none"
        ):
            transformed = self._apply_value_transforms_to_outputs(out_heads)
            flat = transformed.transpose(1, 2).contiguous().view(
                batch, seq_len, self.num_heads * self.value_dim
            )
            return self.out_proj(flat)

        transforms = self.compute_value_transforms()
        weight_blocks = self.out_proj.weight.view(
            self.d_model, self.num_heads, self.value_dim
        )
        effective_weight = torch.einsum(
            "hde,ohe->ohd", transforms, weight_blocks
        ).reshape(self.d_model, self.num_heads * self.value_dim)
        flat = out_heads.transpose(1, 2).contiguous().view(
            batch, seq_len, self.num_heads * self.value_dim
        )
        return F.linear(flat, effective_weight, self.out_proj.bias)

    def _expand_keys(self, k: torch.Tensor) -> torch.Tensor:
        if k.ndim == 3:
            keys = self._reshape_base_qk(k)
        elif k.ndim == 4:
            keys = k
        else:
            raise ValueError("keys must have shape [batch, time, bases*dim] or [batch, bases, time, dim]")
        batch, _, seq_len, _ = keys.shape
        keys = keys[:, :, None, :, :].expand(
            batch,
            self.num_base_heads,
            self.generated_heads_per_base,
            seq_len,
            self.base_dim,
        )
        return keys.reshape(batch, self.num_heads, seq_len, self.base_dim)

    def _prepare_additive_mask(
        self,
        scores: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        neg_large = _negative_large(scores.dtype)
        batch, _, target_len, source_len = scores.shape

        if self.causal:
            causal_mask = torch.ones(
                target_len, source_len, device=scores.device, dtype=torch.bool
            ).triu(source_len - target_len + 1)
            scores = scores.masked_fill(causal_mask[None, None, :, :], neg_large)

        if attn_mask is not None:
            attn_mask = attn_mask.to(device=scores.device)
            if attn_mask.dtype == torch.bool:
                if attn_mask.ndim == 2:
                    scores = scores.masked_fill(attn_mask[None, None, :, :], neg_large)
                elif attn_mask.ndim == 3:
                    scores = scores.masked_fill(attn_mask[:, None, :, :], neg_large)
                elif attn_mask.ndim == 4:
                    scores = scores.masked_fill(attn_mask, neg_large)
                else:
                    raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
            else:
                scores = scores + self._broadcast_attn_mask(attn_mask, batch)

        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, source_len):
                raise ValueError(
                    "key_padding_mask must have shape (batch, source_len), got "
                    f"{tuple(key_padding_mask.shape)}"
                )
            mask = key_padding_mask.to(device=scores.device, dtype=torch.bool)
            scores = scores.masked_fill(mask[:, None, None, :], neg_large)

        return scores

    @staticmethod
    def _broadcast_attn_mask(attn_mask: torch.Tensor, batch: int) -> torch.Tensor:
        if attn_mask.ndim == 2:
            return attn_mask[None, None, :, :]
        if attn_mask.ndim == 3:
            if attn_mask.shape[0] == batch:
                return attn_mask[:, None, :, :]
            return attn_mask[:, :, :, None].squeeze(-1)
        if attn_mask.ndim == 4:
            return attn_mask
        raise ValueError("attn_mask must have 2, 3, or 4 dimensions")

    def _explicit_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = self.compute_scores(q, k, apply_scale=True)
        scores = self._prepare_additive_mask(scores, attn_mask, key_padding_mask)
        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        out_heads = torch.einsum("bhts,bhsd->bhtd", attn, self._expand_values(v))
        return out_heads, attn

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, target_len, _ = q.shape
        metrics = self.compute_metrics() if self.logit_scale_mode == "rms_metric" else None
        q_metric = self._apply_metric_to_queries(q, metrics=metrics)
        scale = self._score_scale(metrics=metrics)
        if isinstance(scale, torch.Tensor):
            q_metric = q_metric / scale[None, :, None, None].to(
                device=q_metric.device, dtype=q_metric.dtype
            )
            sdpa_scale = 1.0
        else:
            sdpa_scale = 1.0 / scale
        k_base = self._reshape_base_qk(k) if k.ndim == 3 else k
        v_base = self._reshape_base_values(v) if v.ndim == 3 else v
        source_len = k_base.shape[-2]

        sdpa_mask = self._prepare_sdpa_mask(attn_mask, q.dtype, q.device)
        is_causal = self.causal and target_len == source_len and sdpa_mask is None
        if self.causal and target_len != source_len:
            blocked = torch.ones(
                target_len,
                source_len,
                device=q.device,
                dtype=torch.bool,
            ).triu(source_len - target_len + 1)
            causal_additive = torch.zeros(
                target_len,
                source_len,
                device=q.device,
                dtype=q.dtype,
            ).masked_fill(blocked, _negative_large(q.dtype))
            if sdpa_mask is None:
                sdpa_mask = causal_additive
            else:
                sdpa_mask = sdpa_mask + causal_additive
        if key_padding_mask is not None:
            neg_large = _negative_large(q.dtype)
            pad_mask = key_padding_mask[:, None, None, :].to(device=q.device, dtype=torch.bool)
            pad_additive = torch.zeros(
                batch, 1, 1, source_len, device=q.device, dtype=q.dtype
            ).masked_fill(pad_mask, neg_large)
            if sdpa_mask is None:
                sdpa_mask = pad_additive
            else:
                sdpa_mask = sdpa_mask + pad_additive

        if not hasattr(F, "scaled_dot_product_attention"):
            out_heads, _ = self._explicit_attention(q, k, v, attn_mask, key_padding_mask)
            return out_heads

        dropout_p = self.dropout if self.training else 0.0
        if (
            self.num_heads == self.num_base_heads
            and self.num_heads == self.num_value_base_heads
        ):
            out_heads = F.scaled_dot_product_attention(
                q_metric,
                k_base,
                v_base,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=sdpa_scale,
            )
        elif (
            self.sdpa_gqa_mode == "expand"
            or self.num_base_heads != self.num_value_base_heads
        ):
            out_heads = F.scaled_dot_product_attention(
                q_metric,
                self._expand_keys(k_base),
                self._expand_values(v_base),
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=sdpa_scale,
            )
        elif self.sdpa_gqa_mode == "native":
            out_heads = F.scaled_dot_product_attention(
                q_metric,
                k_base,
                v_base,
                attn_mask=sdpa_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=sdpa_scale,
                enable_gqa=True,
            )
        else:
            try:
                out_heads = F.scaled_dot_product_attention(
                    q_metric,
                    k_base,
                    v_base,
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=sdpa_scale,
                    enable_gqa=True,
                )
            except (TypeError, RuntimeError):
                # Older PyTorch builds lack native GQA. Expand only for the
                # attention call; the persistent cache remains base-sized.
                out_heads = F.scaled_dot_product_attention(
                    q_metric,
                    self._expand_keys(k_base),
                    self._expand_values(v_base),
                    attn_mask=sdpa_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=sdpa_scale,
                )
        return out_heads

    @staticmethod
    def _prepare_sdpa_mask(
        attn_mask: torch.Tensor | None,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if attn_mask is None:
            return None
        mask = attn_mask.to(device=device)
        if mask.dtype == torch.bool:
            neg_large = _negative_large(dtype)
            if mask.ndim == 2:
                mask = mask[None, None, :, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            elif mask.ndim != 4:
                raise ValueError("attn_mask must have 2, 3, or 4 dimensions")
            return torch.zeros_like(mask, dtype=dtype).masked_fill(mask, neg_large)
        mask = mask.to(dtype=dtype)
        if mask.ndim == 2:
            return mask[None, None, :, :]
        if mask.ndim == 3:
            return mask[:, None, :, :]
        if mask.ndim == 4:
            return mask
        raise ValueError("attn_mask must have 2, 3, or 4 dimensions")

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
        context: torch.Tensor | None = None,
    ):
        if x.ndim != 3:
            raise ValueError(f"x must have shape (batch, seq_len, d_model), got {tuple(x.shape)}")

        if context is not None and context.ndim != 3:
            raise ValueError(
                f"context must have shape (batch, source_len, d_model), got {tuple(context.shape)}"
            )
        q, k_new, v_new = self._project(x, context=context)
        k_base_new = self._reshape_base_qk(k_new)
        v_base_new = self._reshape_base_values(v_new)
        k, v = _append_base_cache(k_base_new, v_base_new, past_key_value)
        present_key_value = (k, v)
        if self.use_sdpa and not need_weights:
            out_heads = self._sdpa_attention(q, k, v, attn_mask, key_padding_mask)
            attn = None
        else:
            out_heads, attn = self._explicit_attention(q, k, v, attn_mask, key_padding_mask)

        out = self._output_projection(out_heads)
        if need_weights and attn is None:
            raise RuntimeError("attention weights unavailable on SDPA path")
        if need_weights and use_cache:
            return out, attn, present_key_value
        if need_weights:
            return out, attn
        if use_cache:
            return out, present_key_value
        return out
