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

from .configuration_florence2 import Florence2Config
from transformers.configuration_utils import PretrainedConfig


class XVLAConfig(PretrainedConfig):
    """
    Configuration class for the **XVLA (Extended Vision-Language-Action)** model.

    This configuration defines all submodules of XVLA in a single place:
      - The visual-language backbone (Florence2)
      - The temporal/action transformer
      - The action/proprio setup
    """

    model_type = "xvla"

    def __init__(
        # === Florence backbone ===
        self,
        florence_config: dict | None = None,

        # === Transformer head ===
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_domains: int = 30,
        len_soft_prompts: int = 32,
        dim_time: int = 32,
        max_len_seq: int = 512,
        use_hetero_proj: bool = False,
        soft_prompt_length: int = 32,
        attention_type: str = "mha",
        gt_mha_num_base_heads: int = 4,
        gt_mha_num_generators: int = 8,

        # === Action & proprio ===
        max_action_dim: int = 20,  # Maximum action dimension for padding (used by "auto" action mode)
        real_action_dim: int = 20,
        num_actions: int = 30,
        action_mode: str = "ee6d",
        use_proprio: bool = True,

        # === Future prediction (optional) ===
        use_future_prediction: bool = False,
        predictor_hidden_dim: int = 1024,
        num_future_tokens: int = 4,
        num_future_samples: int = 4,
        flow_sampling_steps: int = 16,
        future_loss_weight: float = 1.0,
        future_horizon_steps: int = 1,
        future_nll_weight: float = 1.0,
        future_kl_weight: float = 1.0,
        future_latent_dim: int = 256,
        future_hidden_dim: int = 512,
        future_inject_layer_idx: int = -1,
        use_future_contrastive: bool = False,
        contrastive_loss_weight: float = 0.0,
        contrastive_num_layers: int = 3,
        contrastive_proj_dim: int = 256,
        contrastive_temperature: float = 0.07,

        **kwargs,
    ):
        # Florence2 backbone configuration
        if isinstance(florence_config, dict):
            self.florence_config = Florence2Config(**florence_config)
        elif isinstance(florence_config, Florence2Config):
            self.florence_config = florence_config
        else:
            self.florence_config = Florence2Config()

        # Transformer hyperparameters
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.num_domains = num_domains
        self.len_soft_prompts = len_soft_prompts
        self.dim_time = dim_time
        self.max_len_seq = max_len_seq
        self.use_hetero_proj = use_hetero_proj
        self.soft_prompt_length = soft_prompt_length
        if attention_type not in {"mha", "gt_mha_residual"}:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
        if num_heads % gt_mha_num_base_heads:
            raise ValueError("num_heads must be divisible by gt_mha_num_base_heads")
        self.attention_type = attention_type
        self.gt_mha_num_base_heads = gt_mha_num_base_heads
        self.gt_mha_num_generators = gt_mha_num_generators

        # Action/proprioception settings
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.use_proprio = use_proprio
        
        self.real_action_dim = real_action_dim
        self.max_action_dim = max_action_dim

        # Future prediction settings
        self.use_future_prediction = use_future_prediction
        self.predictor_hidden_dim = predictor_hidden_dim
        self.num_future_tokens = num_future_tokens
        self.num_future_samples = num_future_samples
        self.flow_sampling_steps = flow_sampling_steps
        self.future_loss_weight = future_loss_weight
        self.future_horizon_steps = future_horizon_steps
        self.future_nll_weight = future_nll_weight
        self.future_kl_weight = future_kl_weight
        self.future_latent_dim = future_latent_dim
        self.future_hidden_dim = future_hidden_dim
        self.future_inject_layer_idx = future_inject_layer_idx
        self.use_future_contrastive = use_future_contrastive
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_num_layers = contrastive_num_layers
        self.contrastive_proj_dim = contrastive_proj_dim
        self.contrastive_temperature = contrastive_temperature
        
        # Initialize base HF config attributes (e.g. name_or_path)
        super().__init__(**kwargs)

    # -------------------------------------------------------------------------
    # Serialization helpers
    # -------------------------------------------------------------------------
    def to_dict(self):
        """
        Convert this configuration (and its Florence sub-config)
        into a fully serializable dictionary for HF save/load.
        """
        output = super().to_dict()
        output["florence_config"] = self.florence_config.to_dict()
        return output
