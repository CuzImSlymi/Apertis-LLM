import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import numpy as np
import sys
import os
from pathlib import Path
import json
import inspect
from packaging import version

import logging
logger = logging.getLogger(__name__)

# Flash Attention imports
try:
    IS_FLASH_ATTN_AVAILABLE = version.parse(torch.__version__) >= version.parse("2.0.0") and torch.cuda.is_available()
    if IS_FLASH_ATTN_AVAILABLE:
        from flash_attn import flash_attn_func
        # For future use with padding: from flash_attn import flash_attn_varlen_func
    else:
        flash_attn_func = None
except ImportError:
    logger.info("Flash Attention not available. Install flash-attn for potential speedups.")
    flash_attn_func = None
    IS_FLASH_ATTN_AVAILABLE = False


if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.multimodal.module import UnifiedMultimodalEncoder


class ApertisConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 2048,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
        position_embedding_type: str = "rotary",
        use_cache: bool = True,
        classifier_dropout: float = None,
        model_type: str = "apertis",
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        sliding_window: Optional[int] = None,
        attention_type: str = "standard_mha",
        ssm_d_inner: Optional[int] = None,
        ssm_d_state: int = 16,
        ssm_dt_rank: Union[int, str] = "auto",
        ssm_conv_kernel: int = 4,
        use_flash_attention: bool = False,
        use_expert_system: bool = False,
        num_experts: int = 8,
        experts_per_token: int = 2,
        multimodal: bool = False,
        image_size: int = 224,
        vision_embed_dim: int = 768,
        vision_patch_size: int = 16,
        vision_layers: int = 12,
        vision_heads: int = 12,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # MoE specific configs
        load_balancing_loss_coef: float = 0.01,
        expert_capacity_factor: float = 1.25, # Max proportion of tokens an expert can take = (tokens_per_batch / num_experts) * expert_capacity_factor
        noisy_routing_alpha: float = 0.1, # Factor for learnable noise in top-k routing
        expert_dropout_prob: float = 0.1, # Dropout probability for expert outputs during training
        router_z_loss_coef: float = 0.001, # Coefficient for router z-loss (encourages router confidence)
        expert_output_gating: bool = False, # Not implemented in this plan, placeholder
        use_noisy_top_k_routing: bool = True,
        use_expert_capacity_limit: bool = True,
        use_expert_dropout: bool = True, # This refers to dropping entire experts, not dropout within MLP
        use_router_z_loss: bool = True,
        use_load_balancing_loss: bool = True,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.model_type = model_type
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_type = attention_type
        self.ssm_d_state = ssm_d_state
        
        if self.attention_type == "selective_ssm":
            derived_ssm_d_inner = self.num_attention_heads * self.ssm_d_state
            if ssm_d_inner is not None and ssm_d_inner != derived_ssm_d_inner:
                logger.warning(f"For attention_type='selective_ssm', ssm_d_inner ({ssm_d_inner}) is overridden by num_attention_heads*ssm_d_state ({derived_ssm_d_inner}).")
            self.ssm_d_inner = derived_ssm_d_inner
        elif ssm_d_inner is None:
            self.ssm_d_inner = 2 * self.hidden_size
        else:
            self.ssm_d_inner = ssm_d_inner

        if ssm_dt_rank == "auto":
            self.ssm_dt_rank = math.ceil(self.hidden_size / 16)
        else:
            self.ssm_dt_rank = int(ssm_dt_rank)
        self.ssm_conv_kernel = ssm_conv_kernel
        
        self.use_flash_attention = use_flash_attention
        self.use_expert_system = use_expert_system
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.multimodal = multimodal
        self.image_size = image_size
        self.vision_embed_dim = vision_embed_dim
        self.vision_patch_size = vision_patch_size
        self.vision_layers = vision_layers
        self.vision_heads = vision_heads
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states

        # MoE attributes
        self.load_balancing_loss_coef = load_balancing_loss_coef
        self.expert_capacity_factor = expert_capacity_factor
        self.noisy_routing_alpha = noisy_routing_alpha
        self.expert_dropout_prob = expert_dropout_prob
        self.router_z_loss_coef = router_z_loss_coef
        self.expert_output_gating = expert_output_gating # Placeholder
        self.use_noisy_top_k_routing = use_noisy_top_k_routing
        self.use_expert_capacity_limit = use_expert_capacity_limit
        self.use_expert_dropout = use_expert_dropout
        self.use_router_z_loss = use_router_z_loss
        self.use_load_balancing_loss = use_load_balancing_loss

        # Make sure num_experts and experts_per_token are consistent
        if not self.use_expert_system:
            self.num_experts = 0 # Or 1, but 0 indicates no MoE specific logic path
            self.experts_per_token = 0
        else:
            self.experts_per_token = min(self.num_experts, self.experts_per_token) if self.num_experts > 0 else 0


        for key, value in kwargs.items():
            if not hasattr(self, key):
                logger.warning(f"Ignoring unknown config parameter: {key}={value}")


    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        sig = inspect.signature(cls.__init__)
        valid_keys = {param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD or param.kind == param.VAR_KEYWORD}
        
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys or k == "kwargs"}
        
        if "ssm_dt_rank" in filtered_config_dict and filtered_config_dict["ssm_dt_rank"] == "auto":
            hs = filtered_config_dict.get("hidden_size", 768) # Default if hidden_size not in dict
            filtered_config_dict["ssm_dt_rank"] = math.ceil(hs / 16)
        
        return cls(**filtered_config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        config_file_path_str = ""
        if os.path.isdir(model_name_or_path):
            config_file_path_str = os.path.join(model_name_or_path, "config.json")
        elif os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".json"):
            config_file_path_str = model_name_or_path
        else: # Try as directory even if it doesn't exist yet, might be HF hub name
            config_file_path_str = os.path.join(model_name_or_path, "config.json")

        if not os.path.exists(config_file_path_str) and os.path.isdir(model_name_or_path):
            # Fallback: if path is a dir and config.json is not in it, check parent.
            # This is useful if model_path is 'models/my_model/pytorch_model.bin' and config is in 'models/my_model/'
            parent_dir_config_file = os.path.join(Path(model_name_or_path).parent, "config.json")
            if os.path.exists(parent_dir_config_file):
                config_file_path_str = parent_dir_config_file
        
        if not os.path.exists(config_file_path_str):
            raise FileNotFoundError(f"Config file not found. Looked for: '{config_file_path_str}' based on input '{model_name_or_path}'")

        with open(config_file_path_str, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dimension must be even, got {dim}")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        t = torch.arange(self.max_position_embeddings, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        cos_cached = freqs.cos()
        sin_cached = freqs.sin()
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None):
        batch_size, seq_len, _ = x.shape

        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        
        cos = self.cos_cached[position_ids]
        sin = self.sin_cached[position_ids]

        x_reshaped = x.float().reshape(batch_size, seq_len, -1, 2)
        
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        
        rotated_x_part1 = x1 * cos - x2 * sin
        rotated_x_part2 = x1 * sin + x2 * cos
        
        x_out = torch.stack((rotated_x_part1, rotated_x_part2), dim=-1).reshape(batch_size, seq_len, self.dim)
        return x_out.type_as(x)

class SelectiveLinearAttention(nn.Module):
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.d_state = config.ssm_d_state
        self.d_inner = self.num_heads * self.d_state
        self.dt_rank = config.ssm_dt_rank
        self.conv_kernel_size = config.ssm_conv_kernel
        self.head_d_inner_effectively_N = self.d_state
        
        self.in_proj_x = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=self.conv_kernel_size, groups=self.d_inner,
            padding=(self.conv_kernel_size - 1)
        )
        self.x_param_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.num_heads * self.d_state, bias=False)
        self.dt_proj_head = nn.Linear(self.dt_rank, self.num_heads, bias=True)
        torch.nn.init.uniform_(self.dt_proj_head.bias, a=math.log(1e-3), b=math.log(1e-2))
        self.A_log = nn.Parameter(torch.empty(self.num_heads, self.d_state))
        torch.nn.init.uniform_(self.A_log, a=math.log(0.5), b=math.log(0.99))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.hidden_size, bias=False)
        self.use_cache = False
        self.conv_state = None
        self.ssm_state = None

    def _ssm_scan_parallel(self, u_input, delta_for_A, A_log_resolved, B_term, C_modulator):
        B, H, L, N = u_input.shape
        A_cont_diag = torch.exp(A_log_resolved).neg()
        Ab = torch.exp(delta_for_A * A_cont_diag.unsqueeze(0).unsqueeze(2))
        log_Ab = torch.log(Ab + 1e-38)
        cumlog_Ab_inclusive = torch.cumsum(log_Ab, dim=2)
        P_inclusive = torch.exp(cumlog_Ab_inclusive)
        B_div_P_inclusive = B_term / (P_inclusive + 1e-38)
        sum_B_div_P_inclusive = torch.cumsum(B_div_P_inclusive, dim=2)
        h_states = P_inclusive * sum_B_div_P_inclusive
        y_ssm_states = C_modulator * h_states
        return y_ssm_states

    def _ssm_pytorch_scan_recurrent(self, u_scan_input, delta, A_log_resolved, B_term_effective, C_modulator, ssm_state_prev=None):
        B_b, H_b, L_b, N_b = u_scan_input.shape
        A_cont_diag = torch.exp(A_log_resolved).neg()
        delta_A = delta * A_cont_diag.unsqueeze(0).unsqueeze(2)
        A_bar = torch.exp(delta_A)
        if self.use_cache and ssm_state_prev is not None:
            h = ssm_state_prev
        else:
            h = torch.zeros(B_b, H_b, N_b, device=u_scan_input.device, dtype=u_scan_input.dtype)
        ys = []
        for i in range(L_b):
            h = A_bar[:,:,i,:] * h + B_term_effective[:,:,i,:]
            ys.append(C_modulator[:,:,i,:] * h)
        y_stacked = torch.stack(ys, dim=2)
        new_ssm_state = h.detach() if self.use_cache else None
        if self.use_cache: return y_stacked, new_ssm_state
        else: return y_stacked

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False, use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        self.use_cache = use_cache
        B, L, D_hidden = hidden_states.shape
        conv_state_prev, ssm_state_prev = None, None
        if past_key_value is not None:
            conv_state_prev, ssm_state_prev = past_key_value
        x_projected = self.in_proj_x(hidden_states)
        z_gate_features = self.in_proj_z(hidden_states)
        x_conv_input = x_projected.transpose(1, 2)
        if conv_state_prev is not None and use_cache:
            if conv_state_prev.shape[1] == self.d_inner and conv_state_prev.shape[2] == self.conv_kernel_size -1 :
                 x_conv_input = torch.cat([conv_state_prev, x_conv_input], dim=2)
        current_conv_state = x_conv_input[:, :, -(self.conv_kernel_size - 1):].detach() if use_cache else None
        x_convolved = self.conv1d(x_conv_input)[:, :, :L]
        x_convolved = x_convolved.transpose(1, 2)
        x_activated = F.silu(x_convolved)
        ssm_params_raw = self.x_param_proj(x_activated)
        dt_rank_feats, B_raw_params, C_raw_params = torch.split(
            ssm_params_raw,
            [self.dt_rank, self.num_heads * self.d_state, self.num_heads * self.d_state],
            dim=-1
        )
        delta_logits = self.dt_proj_head(dt_rank_feats)
        delta_for_scan = F.softplus(delta_logits).transpose(1,2).unsqueeze(-1)
        B_term_eff = B_raw_params.view(B, L, self.num_heads, self.d_state).transpose(1,2)
        C_mod = C_raw_params.view(B, L, self.num_heads, self.d_state).transpose(1,2)
        u_for_scan = x_activated.view(B, L, self.num_heads, self.d_state).transpose(1,2)
        y_ssm_scan_output, current_ssm_h_state = None, None
        if self.training and not self.use_cache:
            y_ssm_scan_output = self._ssm_scan_parallel(u_for_scan, delta_for_scan, self.A_log, B_term_eff, C_mod)
        else:
            scan_output_tuple = self._ssm_pytorch_scan_recurrent(u_for_scan, delta_for_scan, self.A_log, B_term_eff, C_mod, ssm_state_prev)
            if use_cache: y_ssm_scan_output, current_ssm_h_state = scan_output_tuple
            else: y_ssm_scan_output = scan_output_tuple
        y_ssm_processed = y_ssm_scan_output.transpose(1,2).contiguous().view(B, L, self.d_inner)
        y_ssm_plus_skip = y_ssm_processed + self.D.unsqueeze(0).unsqueeze(0) * x_activated
        output_gated_with_D = y_ssm_plus_skip * F.silu(z_gate_features)
        final_output = self.out_proj(output_gated_with_D)
        current_cache_state = None
        if use_cache:
            current_cache_state = (current_conv_state, current_ssm_h_state)
        return final_output, y_ssm_processed if output_attentions else None, current_cache_state

class AdaptiveExpertSystem(nn.Module):
    def __init__(self, config: ApertisConfig, activation_function_override: Optional[str] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token

        if self.num_experts <= 0:
            logger.info("AdaptiveExpertSystem initialized with num_experts <= 0. It will act as a passthrough or be effectively disabled if part of FFN.")
            self.router = None
            self.experts = None
            # Ensure feature flags are False so forward pass doesn't error if called.
            self.use_noisy_top_k_routing = False
            self.use_expert_capacity_limit = False
            self.use_expert_dropout = False
            self.use_router_z_loss = False
            self.use_load_balancing_loss = False
            # Assign default coefs even if not used, for consistency, or handle in forward pass.
            self.load_balancing_loss_coef = 0.0
            self.router_z_loss_coef = 0.0
            self.expert_dropout_prob = 0.0
            self.noisy_routing_alpha = 0.0
            return

        self.router_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.router = nn.Linear(self.hidden_size, self.num_experts)

        expert_activation_fn_str = activation_function_override if activation_function_override is not None else config.hidden_act

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps),
                nn.Linear(self.hidden_size, self.intermediate_size),
                self._get_activation_fn(expert_activation_fn_str),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(self.intermediate_size, self.hidden_size)
            ) for _ in range(self.num_experts)
        ])

        self.w_noise = None
        if config.use_noisy_top_k_routing:
            # Per-expert noise scaling factor for router logits noise. Initialized to zeros.
            self.w_noise = nn.Parameter(torch.zeros(self.num_experts))
            # Alternative: nn.Parameter(torch.full((self.num_experts,), 1e-3)) for small initial noise

        # Store relevant config values from ApertisConfig for use in forward()
        self.load_balancing_loss_coef = config.load_balancing_loss_coef
        self.expert_capacity_factor = config.expert_capacity_factor
        self.router_z_loss_coef = config.router_z_loss_coef
        self.noisy_routing_alpha = config.noisy_routing_alpha
        self.expert_dropout_prob = config.expert_dropout_prob # Used for expert masking during training

        self.use_noisy_top_k_routing = config.use_noisy_top_k_routing
        self.use_expert_capacity_limit = config.use_expert_capacity_limit
        self.use_expert_dropout = config.use_expert_dropout # This is for dropping entire experts
        self.use_router_z_loss = config.use_router_z_loss
        self.use_load_balancing_loss = config.use_load_balancing_loss
        
    def _get_activation_fn(self, act_fn_str: str):
        if act_fn_str == "gelu": return nn.GELU()
        elif act_fn_str == "relu": return nn.ReLU()
        elif act_fn_str == "silu" or act_fn_str == "swish": return nn.SiLU()
        logger.warning(f"Unsupported activation: {act_fn_str} in AdaptiveExpertSystem. Defaulting to GELU.")
        return nn.GELU()
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lb_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)
        rz_loss = torch.tensor(0.0, device=hidden_states.device, dtype=hidden_states.dtype)

        if self.num_experts <= 0 or self.experts is None or self.router is None:
            return hidden_states, lb_loss, rz_loss

        B, L, D = hidden_states.shape
        S = B * L  # Total number of tokens in batch

        flat_hidden_states = hidden_states.reshape(S, D)
        normalized_hidden = self.router_norm(flat_hidden_states)
        router_logits_flat = self.router(normalized_hidden).float()  # Shape: (S, num_experts)

        # Step 3: Noisy Top-K Routing
        if self.use_noisy_top_k_routing and self.training:
            noise_scale = F.softplus(self.w_noise) * self.noisy_routing_alpha
            noise = torch.randn_like(router_logits_flat) * noise_scale.unsqueeze(0) # noise_scale is (E,)
            router_logits_flat = router_logits_flat + noise

        # Gating probabilities and top-k selection
        gates_all_experts = F.softmax(router_logits_flat, dim=-1) # (S, num_experts)
        routing_gate_probs, selected_indices = torch.topk(gates_all_experts, self.experts_per_token, dim=-1) # (S, K)

        # Step 4: Load Balancing Loss
        # P_i: Mean probability mass for each expert.
        # f_i: Fraction of tokens selecting expert 'i' among top_k (proxy before capacity).
        # For a more accurate f_i after capacity, we'd need actual dispatch counts.
        # Let's keep the current proxy f_i for lb_loss, as actual counts are complex before dispatch.
        if self.use_load_balancing_loss and self.training and self.load_balancing_loss_coef > 0:
            P_i = torch.mean(gates_all_experts, dim=0) # (num_experts,)
            # expert_selection_one_hot: (S, num_experts), 1 if expert is in top_k for token s
            expert_selection_one_hot = torch.zeros_like(gates_all_experts)
            expert_selection_one_hot.scatter_(1, selected_indices, 1)
            f_i = torch.mean(expert_selection_one_hot, dim=0) # (num_experts,)
            lb_loss = self.load_balancing_loss_coef * self.num_experts * torch.sum(f_i * P_i)

        # Step 5: Expert Capacity Calculation
        expert_capacity = S # Default to no capacity limit if not training or not enabled
        if self.use_expert_capacity_limit and self.training and self.num_experts > 0:
            calculated_capacity = math.floor((S / self.num_experts) * self.expert_capacity_factor)
            expert_capacity = max(1, calculated_capacity) if S > 0 else 0

        # Step 6: Expert Dropout Mask
        active_expert_mask = torch.ones(self.num_experts, device=hidden_states.device, dtype=torch.bool)
        if self.use_expert_dropout and self.training and self.expert_dropout_prob > 0 and self.num_experts > 0:
            num_to_drop = math.floor(self.num_experts * self.expert_dropout_prob)
            if num_to_drop >= self.num_experts : num_to_drop = self.num_experts -1 # Ensure at least one expert is active
            if num_to_drop > 0:
                perm = torch.randperm(self.num_experts, device=hidden_states.device)
                dropped_indices = perm[:num_to_drop]
                active_expert_mask[dropped_indices] = False

        # Step 7: Router Z-Loss
        if self.use_router_z_loss and self.training and self.router_z_loss_coef > 0:
            log_z = torch.logsumexp(router_logits_flat, dim=-1)  # (S,)
            rz_loss = self.router_z_loss_coef * torch.mean(log_z**2)

        # Normalize routing_gate_probs for combining expert outputs
        routing_weights = routing_gate_probs / (torch.sum(routing_gate_probs, dim=-1, keepdim=True) + 1e-6) # (S,K)

        final_hidden_states = torch.zeros_like(flat_hidden_states)
        expert_token_counts_post_capacity = torch.zeros(self.num_experts, device=hidden_states.device, dtype=torch.long)

        # Efficient Dispatch:
        # Combine tokens for each expert, apply capacity, then process.
        # This involves creating a flat list of (token_idx, expert_idx, weight_for_that_expert)
        # and then batching per expert.

        # For each token, iterate through its K choices.
        # If an expert is chosen, active, and has capacity, process the token with that expert.
        # This loop still iterates K times, but the inner part is more vectorized.

        # Create a mask for tokens that have been dispatched to at least one expert successfully
        # This is to handle cases where a token might be dropped by all its K choices due to capacity/dropout.
        # For now, such tokens will result in zeros. The problem asks for "graceful" drop (return zeros).

        for k_choice_idx in range(self.experts_per_token):
            expert_indices_k = selected_indices[:, k_choice_idx] # (S,) - expert chosen for each token at k-th choice
            gate_weights_k = routing_weights[:, k_choice_idx]    # (S,) - weight for this k-th choice

            for expert_j in range(self.num_experts):
                if not active_expert_mask[expert_j]: # Expert is dropped out
                    continue

                # Mask for tokens selecting expert_j as their k-th choice
                token_mask_for_expert_j_k_choice = (expert_indices_k == expert_j)
                if not token_mask_for_expert_j_k_choice.any():
                    continue

                # Tokens (indices) that selected expert_j at this k-th slot
                candidate_token_indices = token_mask_for_expert_j_k_choice.nonzero(as_tuple=True)[0]

                # Apply capacity
                num_candidate_tokens = candidate_token_indices.shape[0]
                current_expert_load = expert_token_counts_post_capacity[expert_j]

                # How many more tokens can this expert take?
                remaining_capacity_for_expert_j = expert_capacity - current_expert_load

                if remaining_capacity_for_expert_j <= 0 and self.use_expert_capacity_limit and self.training : # Expert is full
                    continue

                # Number of tokens we can actually assign to this expert from this k-th choice slot
                num_to_assign = num_candidate_tokens
                if self.use_expert_capacity_limit and self.training:
                     num_to_assign = min(num_candidate_tokens, remaining_capacity_for_expert_j)

                if num_to_assign < num_candidate_tokens: # Overflow for this (expert_j, k_choice_idx) pair
                    # Select tokens with highest gate_weights_k for this expert
                    overflow_gate_weights = gate_weights_k[candidate_token_indices]
                    _, top_indices_within_candidates = torch.topk(overflow_gate_weights, num_to_assign)
                    actual_token_indices_to_process = candidate_token_indices[top_indices_within_candidates]
                else:
                    actual_token_indices_to_process = candidate_token_indices[:num_to_assign] # Select all or up to num_to_assign

                if actual_token_indices_to_process.shape[0] == 0:
                    continue

                # Update load for this expert
                expert_token_counts_post_capacity[expert_j] += actual_token_indices_to_process.shape[0]

                # Gather tokens and their specific weights for this expert and this k-th choice
                selected_tokens_for_processing = flat_hidden_states[actual_token_indices_to_process]
                weights_for_processing = gate_weights_k[actual_token_indices_to_process].unsqueeze(1) # (num_selected, 1)

                expert_output = self.experts[expert_j](selected_tokens_for_processing)

                # Scatter-add to final output
                # Using index_add_ for non-overlapping updates if a token is assigned to multiple k choices
                # but here each (token, k_choice) is unique.
                # If a token is selected for expert_j via its k_choice_idx, its output is added.
                # This correctly sums contributions if a token is routed to multiple *different* experts via its K choices.
                # If a token could be routed to the *same* expert via multiple K choices (not typical with topk),
                # this simple += would be fine.
                final_hidden_states.index_add_(0, actual_token_indices_to_process, expert_output * weights_for_processing)

        return final_hidden_states.reshape(B, L, D), lb_loss, rz_loss

class StateTrackingRecurrentCell(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_norm = nn.LayerNorm(hidden_size)
        self.state_norm = nn.LayerNorm(hidden_size)
        self.update_gate_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.reset_gate_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.candidate_fc = nn.Linear(2 * hidden_size, hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x_seq: torch.Tensor, h_prev: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D_ = x_seq.shape
        if h_prev is None: h_prev = torch.zeros(B, D_, device=x_seq.device, dtype=x_seq.dtype)
        outputs = []
        h_t = h_prev
        for t in range(L):
            x_t = self.input_norm(x_seq[:, t, :])
            h_prev_norm_t = self.state_norm(h_t)
            combined_zur = torch.cat([x_t, h_prev_norm_t], dim=1)
            z_t = torch.sigmoid(self.update_gate_fc(combined_zur))
            r_t = torch.sigmoid(self.reset_gate_fc(combined_zur))
            combined_hcand = torch.cat([x_t, r_t * h_prev_norm_t], dim=1)
            h_cand_t = torch.tanh(self.candidate_fc(combined_hcand))
            h_t = (1 - z_t) * h_t + z_t * h_cand_t
            h_t = self.dropout(h_t)
            outputs.append(self.output_layer_norm(h_t))
        return torch.stack(outputs, dim=1), h_t

class ApertisAttention(nn.Module):
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = self.hidden_size // self.num_attention_heads
        
        if config.attention_type == "selective_ssm" or config.attention_type == "selective_linear":
            if config.attention_type == "selective_linear" and config.attention_type != "selective_ssm":
                 logger.warning("Config: 'selective_linear' is now treated as 'selective_ssm'.")
            self.attention_mechanism_impl = SelectiveLinearAttention(config=config)
        elif config.attention_type == "standard_mha":
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.attention_mechanism_impl = None
        else:
            logger.error(f"Unsupported attention_type '{config.attention_type}'. Defaulting to 'standard_mha'.")
            self.config.attention_type = "standard_mha"
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_probs_dropout_prob == 0.0)
            self.attention_mechanism_impl = None

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.rope = None
        # Initialize RoPE here if standard MHA is used and RoPE is configured.
        # This RoPE will be applied *after* Q/K projections.
        if config.position_embedding_type == "rotary" and config.attention_type == "standard_mha":
            # The RoPE dimension should match the dimension of Q and K *before* they are split into heads,
            # which is config.hidden_size as q_proj and k_proj map hidden_size to hidden_size.
            self.rope = RotaryEmbedding(
                dim=config.hidden_size, # Applied to Q/K of hidden_size
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta
            )

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_s: torch.Tensor, att_mask: Optional[torch.Tensor]=None,
                pos_ids: Optional[torch.Tensor]=None, past_kv: Optional[Any]=None,
                output_att: bool=False, use_c: bool=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        residual = hidden_s
        normed_hidden_s = self.pre_norm(hidden_s)
        
        att_out, att_weights_proxy, present_cache = None, None, None

        if self.config.attention_type == "selective_ssm":
            att_out, att_weights_proxy, present_cache = self.attention_mechanism_impl(
                normed_hidden_s, attention_mask=att_mask,
                position_ids=pos_ids, past_key_value=past_kv,
                output_attentions=output_att, use_cache=use_c
            )
        elif self.config.attention_type == "standard_mha":
            B, L_q_curr, D = normed_hidden_s.shape
            
            query_states = self.q_proj(normed_hidden_s)
            key_states = self.k_proj(normed_hidden_s)
            value_states = self.v_proj(normed_hidden_s)

            # Apply RoPE if configured and available
            if self.rope is not None:
                # pos_ids are passed from ApertisLayer forward method
                query_states = self.rope(query_states, position_ids=pos_ids)
                key_states = self.rope(key_states, position_ids=pos_ids)

            # Flash Attention V2 path
            # Conditions for using Flash Attention:
            # 1. Flash Attention is available (imported successfully)
            # 2. `use_flash_attention` is True in config
            # 3. Not using KV cache (`use_c` is False) - FlashAttention is mainly for training.
            # 4. Attention weights are not requested (`output_att` is False) - FlashAttention does not return them.
            # 5. `att_mask` is None. FlashAttention's `causal=True` handles decoder masking.
            #    If `att_mask` is present, it implies padding or other custom masking, which
            #    `flash_attn_func` with `causal=True` doesn't handle.
            #    (Step 5 of the plan will look into `_prepare_decoder_attention_mask` interactions)
            # 6. RoPE is handled at ApertisLayer, so q,k here are already rotated if needed.
            #    The RoPE application point will be reviewed in Step 4.

            can_use_flash = (
                IS_FLASH_ATTN_AVAILABLE and
                flash_attn_func is not None and
                self.config.use_flash_attention and
                not use_c and
                not output_att and
                att_mask is None # Crucial: only allow if no explicit mask beyond causal is needed
                                 # This implies inputs are expected to be unpadded or handled before this point
                                 # if flash_attn_varlen_func were to be used.
            )

            if can_use_flash:
                # Reshape Q, K, V for FlashAttention: (batch_size, seqlen, nheads, head_dim)
                # Current _transpose_for_scores gives (B, H, L, Dh)
                # FlashAttention's flash_attn_func expects (B, L, H, Dh)
                query_layer_flash = self._transpose_for_scores(query_states).permute(0, 2, 1, 3)
                key_layer_flash = self._transpose_for_scores(key_states).permute(0, 2, 1, 3)
                value_layer_flash = self._transpose_for_scores(value_states).permute(0, 2, 1, 3)

                # Ensure head_dim is compatible (e.g. divisible by 8, up to 128 for FA v2)
                # This should be generally true for typical LLM configs.
                # ApertisConfig ensures hidden_size is divisible by num_attention_heads.

                context_layer = flash_attn_func(
                    query_layer_flash, key_layer_flash, value_layer_flash,
                    dropout_p=self.attention_dropout.p if self.training else 0.0,
                    softmax_scale=None,  # Defaults to 1/sqrt(head_dim)
                    causal=True          # Handles causal masking for decoder-style models
                )

                # Reshape output back to (B, L, D_hidden)
                # context_layer is (B, L, H, Dh)
                context_layer = context_layer.reshape(B, L_q_curr, self.hidden_size)
                att_out = self.out_proj(context_layer)
                att_weights_proxy = None # FlashAttention does not return attention weights
                present_cache = None     # Not used with FlashAttention path here

            else: # Standard PyTorch Multi-Head Attention path
                if use_c and past_kv is not None:
                    # past_kv[0] is key_states, past_kv[1] is value_states from previous steps
                    # key_states and value_states are (B, L_prev, D) before transpose, (B, H, L_prev, Dh) after
                    # The q_proj, k_proj, v_proj produce (B, L_q_curr, D)
                    # We need to ensure shapes are compatible for cat.
                    # The current code structure of _transpose_for_scores handles (B,L,D) -> (B,H,L,Dh)
                    # So, if past_kv stores K,V after projection but before transpose:
                    # past_kv[0] shape: (B, L_past, D_total_kv)
                    # key_states shape: (B, L_q_curr, D_total_kv)
                    # This seems to be the intention as past_kv is set from key_states, value_states
                    # *before* they are transposed by _transpose_for_scores.
                    # However, the cache is set with key_states, value_states *after* they are computed
                    # from q_proj etc. but *before* _transpose_for_scores.
                    # The shapes are (B, L, D).
                    # So, this cat is correct.
                    key_states = torch.cat([past_kv[0], key_states], dim=1)
                    value_states = torch.cat([past_kv[1], value_states], dim=1)

                present_cache = (key_states, value_states) if use_c else None

                query_layer = self._transpose_for_scores(query_states) # (B, H, L_q, Dh)
                key_layer = self._transpose_for_scores(key_states)     # (B, H, L_kv, Dh)
                value_layer = self._transpose_for_scores(value_states) # (B, H, L_kv, Dh)

                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)

                if att_mask is not None:
                    # att_mask comes from _prepare_decoder_attention_mask (already combined padding and causal)
                    attention_scores = attention_scores + att_mask
                elif L_q_curr > 1: # No att_mask provided (e.g. _prepare_decoder_attention_mask returned None due to no padding)
                                   # AND current query length > 1. We MUST apply a causal mask for standard MHA.
                    q_seq_len = L_q_curr
                    kv_seq_len = key_layer.shape[-2] # This is L_kv_total for the current layer (past + current)

                    # Create boolean causal mask: True where attention is allowed.
                    # query token q_i (index i in current query block) corresponds to absolute position: (kv_seq_len - q_seq_len) + i
                    # key token k_j (index j in full key sequence) corresponds to absolute position: j
                    # Condition for attention: (kv_seq_len - q_seq_len) + i >= j
                    row_idx = torch.arange(q_seq_len, device=query_layer.device).unsqueeze(-1) # Shape (q_seq_len, 1)
                    col_idx = torch.arange(kv_seq_len, device=query_layer.device).unsqueeze(0)  # Shape (1, kv_seq_len)

                    causal_mask_bool = (row_idx + (kv_seq_len - q_seq_len)) >= col_idx # Shape (q_seq_len, kv_seq_len)

                    # Convert to additive float mask matching attention_scores shape for broadcasting
                    # Target shape for addition is (B, H, q_seq_len, kv_seq_len)
                    # causal_mask_bool is (q_seq_len, kv_seq_len)
                    additive_causal_mask = torch.zeros_like(causal_mask_bool, dtype=attention_scores.dtype)
                    additive_causal_mask.masked_fill_(~causal_mask_bool, torch.finfo(attention_scores.dtype).min)

                    attention_scores = attention_scores + additive_causal_mask.unsqueeze(0).unsqueeze(0) # Expand for B, H

            
                attention_probs = nn.functional.softmax(attention_scores, dim=-1)
                att_weights_proxy = attention_probs if output_att else None
            
                attention_probs = self.attention_dropout(attention_probs)
            
                context_layer = torch.matmul(attention_probs, value_layer) # (B, H, L_q, Dh)
                context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (B, L_q, H, Dh)
                new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,) # (B, L_q, D_hidden)
                context_layer = context_layer.view(new_context_layer_shape)
            
                att_out = self.out_proj(context_layer)
        else:
            raise ValueError(f"Unexpected attention_type in ApertisAttention: {self.config.attention_type}")

        dropped_att_out = self.output_dropout(att_out)
        final_out = dropped_att_out + residual
        return final_out, att_weights_proxy, present_cache

class ApertisFeedForward(nn.Module):
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_expert_system and config.num_experts > 0 : # Ensure num_experts is positive
            self.ffn = AdaptiveExpertSystem(
                config=config, # Pass the full config object
                activation_function_override=config.hidden_act # Can be overridden if needed
            )
        else:
            if config.use_expert_system and config.num_experts <= 0:
                logger.warning(f"use_expert_system is True but num_experts is {config.num_experts}. Falling back to standard FFN.")
            self.ffn = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                self._get_activation_fn(config.hidden_act),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.intermediate_size, config.hidden_size),
            )
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def _get_activation_fn(self, act_fn_str: str):
        if act_fn_str == "gelu": return nn.GELU()
        elif act_fn_str == "relu": return nn.ReLU()
        elif act_fn_str == "silu" or act_fn_str == "swish": return nn.SiLU()
        return nn.GELU()
    
    def forward(self, hidden_s: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        residual = hidden_s
        normed_hidden_s = self.pre_norm(hidden_s)

        total_lb_loss = torch.tensor(0.0, device=hidden_s.device, dtype=hidden_s.dtype)
        total_rz_loss = torch.tensor(0.0, device=hidden_s.device, dtype=hidden_s.dtype)

        if isinstance(self.ffn, AdaptiveExpertSystem):
            ffn_out, lb_loss, rz_loss = self.ffn(normed_hidden_s)
            if self.config.use_expert_system: # Only add losses if expert system is truly active
                 total_lb_loss = lb_loss
                 total_rz_loss = rz_loss
        else: # Standard MLP
            ffn_out = self.ffn(normed_hidden_s)
            # lb_loss and rz_loss remain 0.0 as initialized

        dropped_ffn_out = self.output_dropout(ffn_out)
        final_ffn_output = dropped_ffn_out + residual

        # Always return three values.
        # total_lb_loss and total_rz_loss are correctly 0.0 if not updated by a real expert system call.
        return final_ffn_output, total_lb_loss, total_rz_loss

class ApertisLayer(nn.Module):
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.attention = ApertisAttention(config)
        self.feed_forward = ApertisFeedForward(config)
        self.rope = None
        if config.position_embedding_type == "rotary":
            self.rope = RotaryEmbedding(config.hidden_size, config.max_position_embeddings, config.rope_theta)

    def forward(self, hidden_s: torch.Tensor, att_mask: Optional[torch.Tensor]=None,
                pos_ids: Optional[torch.Tensor]=None, past_kv: Optional[Any]=None,
                output_att: bool=False, use_c: bool=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any], torch.Tensor, torch.Tensor]: # Added lb_loss, rz_loss
        
        # RoPE application is now handled inside ApertisAttention for standard_mha
        att_out, att_w, present_att_cache = self.attention(
            hidden_s, att_mask, pos_ids, past_kv, output_att, use_c
        )

        # ApertisFeedForward.forward now returns: final_ffn_output, total_lb_loss, total_rz_loss
        layer_out, lb_loss, rz_loss = self.feed_forward(att_out)

        return layer_out, att_w, present_att_cache, lb_loss, rz_loss

class ApertisModel(nn.Module):
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.abs_pos_embeddings = None
        if config.position_embedding_type == "absolute":
            self.abs_pos_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.multimodal_encoder = None
        self.vision_projection = nn.Identity()
        if config.multimodal:
            self.multimodal_encoder = UnifiedMultimodalEncoder(config)
            if config.vision_embed_dim != config.hidden_size:
                self.vision_projection = nn.Linear(config.vision_embed_dim, config.hidden_size)
        self.layers = nn.ModuleList([ApertisLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_post_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.apply(self._init_weights)
        self.gradient_checkpointing = False
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if hasattr(module, 'bias') and module.bias is not None: module.bias.data.zero_()
            if hasattr(module, 'weight') and module.weight is not None: module.weight.data.fill_(1.0)
        if isinstance(module, SelectiveLinearAttention):
            if hasattr(module, 'dt_proj_head') and module.dt_proj_head.bias is not None:
                 torch.nn.init.uniform_(module.dt_proj_head.bias, a=math.log(1e-3), b=math.log(1e-2))

    def gradient_checkpointing_enable(self): self.gradient_checkpointing = True
    def get_input_embeddings(self): return self.token_embeddings
    def set_input_embeddings(self, embs): self.token_embeddings = embs
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        old_embeddings = self.token_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, self.config.hidden_size, self.padding_idx, device=old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        self._init_weights(new_embeddings)

        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        self.token_embeddings = new_embeddings
        self.config.vocab_size = new_num_tokens # Crucial: Update config
        self.vocab_size = new_num_tokens # Update model attribute if it exists

        if self.padding_idx is not None and self.padding_idx >= new_num_tokens:
            logger.warning(f"Padding idx {self.padding_idx} is out of new vocab size {new_num_tokens}. Setting to 0.")
            self.padding_idx = 0
            self.token_embeddings.padding_idx = self.padding_idx

        logger.info(f"Resized token embeddings from {old_embeddings.num_embeddings} to {new_num_tokens} tokens.")
        return self.token_embeddings

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # attention_mask is the raw padding mask (batch_size, total_sequence_length) from inputs.
        # True for tokens to attend to, False for padding. Could be None if no padding info.

        # Condition for returning None (to enable FlashAttention with causal=True):
        # 1. No padding mask provided explicitly (attention_mask is None).
        # OR 2. A padding mask is provided, but it indicates no tokens are actually padded (all True/1s).
        # This allows FlashAttention to be used for unpadded sequences, including multimodal ones where
        # image tokens (all attend=True) are concatenated with unpadded text tokens.
        use_flash_path_signal = False
        if attention_mask is None:
            use_flash_path_signal = True
        elif torch.all(attention_mask.bool()): # Check if all elements are True (no actual padding that requires masking)
            use_flash_path_signal = True

        if use_flash_path_signal:
            # If conditions for FlashAttention's simple causal masking are met (no actual padding),
            # return None. ApertisAttention will use flash_attn_func(causal=True).
            # The standard MHA path in ApertisAttention will also need to correctly create a
            # causal mask if it receives None and L_q > 1 (this was addressed in a previous step).
            return None

        # If attention_mask (padding mask) is provided AND it contains actual padding (some False/0s),
        # then construct the combined float mask for the standard PyTorch attention path.
        combined_attention_mask = None # This will be a boolean mask: True means attend.
        if input_shape[1] > 1: # L_q > 1, typically training or prefill
            L_q = input_shape[1]
            L_kv = past_key_values_length + L_q
            
            full_causal_mask = torch.ones((L_kv, L_kv), dtype=torch.bool, device=inputs_embeds.device).tril(diagonal=0)
            causal_mask_for_current_q = full_causal_mask[past_key_values_length:, :]
            causal_mask_for_current_q = causal_mask_for_current_q[None, None, :, :].expand(
                input_shape[0],1, L_q, L_kv
            )

            if attention_mask is not None: 
                padding_mask_expanded = attention_mask[:, None, None, :].expand(
                    input_shape[0], 1, L_q, L_kv
                ).to(torch.bool)
                combined_attention_mask = causal_mask_for_current_q & padding_mask_expanded
            else:
                 combined_attention_mask = causal_mask_for_current_q

        elif past_key_values_length > 0 and attention_mask is not None:
             combined_attention_mask = attention_mask[:, None, None, :].expand(
                input_shape[0], 1, 1, attention_mask.shape[-1]
             ).to(torch.bool)

        if combined_attention_mask is not None:
            float_mask = combined_attention_mask.to(dtype=inputs_embeds.dtype)
            return (1.0 - float_mask) * torch.finfo(inputs_embeds.dtype).min
        return None


    def forward(
        self, input_ids: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.Tensor]=None, past_key_values: Optional[List[Any]]=None,
        inputs_embeds: Optional[torch.Tensor]=None, pixel_values: Optional[torch.Tensor]=None,
        use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None,
        output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[Optional[torch.Tensor], ...]], Optional[List[Any]], Optional[torch.Tensor], Optional[torch.Tensor]]: # Added total_lb_loss, total_rz_loss
        use_c = use_cache if use_cache is not None else self.config.use_cache
        output_att = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hs = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify one of input_ids or inputs_embeds.")
        
        if inputs_embeds is None:
            if input_ids is None: raise ValueError("input_ids required if inputs_embeds is None.")
            inputs_embeds = self.token_embeddings(input_ids)
        
        B, L_curr_query = inputs_embeds.shape[0], inputs_embeds.shape[1]
        
        past_kv_len = 0
        if past_key_values is not None and past_key_values[0] is not None:
            if self.config.attention_type == "standard_mha": # Assuming past_kv[0] is (key_states, value_states)
                past_kv_len = past_key_values[0][0].shape[1] # key_states: (B, L_past, D) or (B,H,L_past,Dh)
            elif self.config.attention_type == "selective_ssm": # Assuming past_kv[0] is (conv_state, ssm_state)
                 # For selective_ssm, past_kv_len is based on the sequence dimension of conv_state or ssm_state.
                 # conv_state is (B, D_inner, L_conv_past)
                 # ssm_state is (B, H, N_ssm_state) - not directly seq len.
                 # The effective sequence length processed by SSM is inferred from conv_state's length.
                 # Let's assume _prepare_decoder_attention_mask correctly uses past_key_values_length = 0 for SSM initial pass
                 # and ApertisLayer handles past_kv for SSM correctly.
                 # For _prepare_decoder_attention_mask, past_kv_len is about text token positions.
                 # If SSM cache exists, it means we are in decoding, L_q_curr = 1.
                 # The `past_key_values_length` for _prepare_decoder_attention_mask needs to be the length of PREVIOUSLY PROCESSED tokens.
                 # This is often tracked via attention_mask.sum() or similar.
                 # For now, let's assume the ApertisLayer's internal cache handling for SSM is independent of this explicit past_kv_len here,
                 # and this past_kv_len is mostly for MHA style KV caching.
                 # For SSM, the "past length" is implicitly handled by its recurrent state.
                 # If past_key_values[0][0] for SSM is conv_state (B, D_inner, L_past_conv_kernel-1), then this is not seq length.
                 # Let's stick to the MHA interpretation for past_kv_len for now as it's more explicit for mask generation.
                 # This might need refinement if _prepare_decoder_attention_mask interacts complexly with SSM's stateful nature.
                if past_key_values[0][0] is not None: # Check if conv_state exists
                    # This is tricky because SSM cache isn't (B, L, D) like MHA.
                    # For _prepare_decoder_attention_mask, `past_key_values_length` should be the number of tokens already processed.
                    # If we are generating token by token, this value increases.
                    # A common way is to use `attention_mask.shape[1] - L_curr_query` if attention_mask covers total length.
                    # Given the current structure, let's assume this `past_kv_len` is correctly determined for MHA.
                    # For SSM, the mask is less critical if it's always causal within its scan.
                    # This definition of past_kv_len might be an oversimplification for SSM if not handled carefully.
                    # However, the current `_prepare_decoder_attention_mask` seems more geared towards MHA.
                    pass # Keep MHA logic for past_kv_len for now
        
        current_pos_ids = position_ids
        if current_pos_ids is None:
            current_pos_ids = torch.arange(past_kv_len, L_curr_query + past_kv_len, dtype=torch.long, device=inputs_embeds.device)
            current_pos_ids = current_pos_ids.unsqueeze(0).expand(B, -1)

        if self.config.position_embedding_type == "absolute" and self.abs_pos_embeddings:
            abs_pos_embeds = self.abs_pos_embeddings(current_pos_ids)
            inputs_embeds = inputs_embeds + abs_pos_embeds
        
        query_embeddings_for_layers = inputs_embeds
        pos_ids_for_layers = current_pos_ids # These are the position_ids for the current query part
        
        num_img_tokens = 0
        if self.config.multimodal and pixel_values is not None and past_kv_len == 0: # Only add image feats at the beginning
            img_feats = self.multimodal_encoder(pixel_values)
            img_feats_proj = self.vision_projection(img_feats)
            num_img_tokens = img_feats_proj.shape[1]

            query_embeddings_for_layers = torch.cat([img_feats_proj, inputs_embeds], dim=1)
            
            # Adjust position_ids for the concatenated sequence
            img_pos_ids = torch.arange(num_img_tokens, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
            # text_pos_ids_shifted should start from num_img_tokens if current_pos_ids originally started from 0 for text
            # If current_pos_ids already reflects past_kv_len, then it should be:
            # current_pos_ids (for text) + num_img_tokens (if text comes after image tokens in sequence)
            # Assuming text tokens follow image tokens conceptually.
            text_pos_ids_shifted = current_pos_ids + num_img_tokens
            pos_ids_for_layers = torch.cat([img_pos_ids, text_pos_ids_shifted], dim=1)
            
            if attention_mask is not None and attention_mask.shape[1] == L_curr_query: # attention_mask is for text part only
                img_padding_mask = torch.ones((B, num_img_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([img_padding_mask, attention_mask], dim=1)
            elif attention_mask is None: # No mask provided, assume all attend
                attention_mask = torch.ones((B, num_img_tokens + L_curr_query), dtype=torch.long, device=query_embeddings_for_layers.device)

        elif self.config.multimodal and pixel_values is not None and past_kv_len > 0:
            logger.warning("pixel_values provided with past_key_values. Ignoring image for this step as it's assumed to be part of the prefix.")

        hidden_s = self.embed_dropout(query_embeddings_for_layers)
        
        # The shape for _prepare_decoder_attention_mask should be based on the final query_embeddings_for_layers
        current_sequence_length = query_embeddings_for_layers.shape[1]
        ext_att_mask = self._prepare_decoder_attention_mask(
            attention_mask, # This mask should now cover the full sequence (image + text if multimodal)
            (B, current_sequence_length),
            query_embeddings_for_layers, # Pass the embeddings that go into layers
            past_kv_len # This past_kv_len is crucial for how causal mask is constructed relative to kv_cache
        )
        
        all_hs_out, all_att_out, all_kv_cache_out = [], [], []
        # Initialize accumulated losses as scalar tensors on the correct device
        accumulated_lb_loss = torch.tensor(0.0, device=hidden_s.device, dtype=hidden_s.dtype)
        accumulated_rz_loss = torch.tensor(0.0, device=hidden_s.device, dtype=hidden_s.dtype)

        for i, layer_mod in enumerate(self.layers):
            if output_hs: all_hs_out.append(hidden_s)
            
            layer_past_kv = past_key_values[i] if past_key_values and i < len(past_key_values) else None
            # Use pos_ids_for_layers which accounts for multimodal prefix for RoPE etc. inside attention
            current_layer_pos_ids_for_attention = pos_ids_for_layers

            layer_lb_loss_iter = torch.tensor(0.0, device=hidden_s.device, dtype=hidden_s.dtype)
            layer_rz_loss_iter = torch.tensor(0.0, device=hidden_s.device, dtype=hidden_s.dtype)

            if self.gradient_checkpointing and self.training and not use_c:
                def create_cp_forward(layer_module_cp):
                    # output_att and use_c are from the outer scope of ApertisModel.forward
                    def _cp_forward(hidden_states_cp, attention_mask_cp, position_ids_cp, past_key_value_cp):
                        return layer_module_cp(hidden_states_cp, attention_mask_cp, position_ids_cp, past_key_value_cp, output_att, use_c)
                    return _cp_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_cp_forward(layer_mod),
                    hidden_s,
                    ext_att_mask, # This is the combined causal and padding mask for the full sequence
                    current_layer_pos_ids_for_attention,
                    layer_past_kv,
                    use_reentrant=False # PyTorch 2.0+ recommendation
                )
                # ApertisLayer.forward returns: layer_out, att_w, present_att_cache, lb_loss, rz_loss
                hidden_s, att_w, present_kv, layer_lb_loss_iter, layer_rz_loss_iter = layer_outputs

                att_w = att_w if output_att else None
                present_kv = present_kv if use_c else None

            else: # Not using gradient checkpointing
                hidden_s, att_w, present_kv, layer_lb_loss_iter, layer_rz_loss_iter = layer_mod(
                    hidden_s, ext_att_mask, current_layer_pos_ids_for_attention, layer_past_kv, output_att, use_c
                )

            if output_att: all_att_out.append(att_w)
            if use_c: all_kv_cache_out.append(present_kv)

            # Accumulate losses if they are valid tensors (not None)
            if self.config.use_expert_system: # Only accumulate if expert system is active
                if layer_lb_loss_iter is not None:
                    accumulated_lb_loss += layer_lb_loss_iter
                if layer_rz_loss_iter is not None:
                    accumulated_rz_loss += layer_rz_loss_iter
        
        hidden_s = self.final_post_norm(hidden_s)
        if output_hs: all_hs_out.append(hidden_s)
        
        final_lb_loss = accumulated_lb_loss if self.config.use_expert_system else None
        final_rz_loss = accumulated_rz_loss if self.config.use_expert_system else None

        return (
            hidden_s,
            tuple(all_hs_out) if output_hs and all_hs_out else None,
            tuple(all_att_out) if output_att and all_att_out else None,
            tuple(all_kv_cache_out) if use_c and all_kv_cache_out else None,
            final_lb_loss,
            final_rz_loss
        )

class ApertisForCausalLM(nn.Module):
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.model = ApertisModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embeddings.weight
        self._init_weights(self.lm_head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module is self.lm_head and not self.config.tie_word_embeddings:
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def load_state_dict(self, state_dict, strict=True):
        return super().load_state_dict(state_dict, strict=strict)
        
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        model_save_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_save_path)
        self.config.save_pretrained(save_directory)
        
    def get_output_embeddings(self): return self.lm_head
    def set_output_embeddings(self, new_lm_head): self.lm_head = new_lm_head

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Linear:
        self.model.resize_token_embeddings(new_num_tokens) # This updates model.config.vocab_size

        if not self.config.tie_word_embeddings or self.lm_head.weight.shape[0] != new_num_tokens :
            old_lm_head = self.lm_head
            self.lm_head = nn.Linear(self.config.hidden_size, new_num_tokens, bias=False,
                                     device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
            self._init_weights(self.lm_head)

            num_tokens_to_copy = min(old_lm_head.out_features, new_num_tokens)
            self.lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
            logger.info(f"Resized (untied or mismatched) LM head from {old_lm_head.out_features} to {new_num_tokens} tokens.")

        elif self.config.tie_word_embeddings:
             self.lm_head.weight = self.model.token_embeddings.weight # Re-tie after ApertisModel resize
             self.lm_head.out_features = new_num_tokens # Manually update out_features for tied head
             logger.info(f"Re-tied LM head. New vocab size: {new_num_tokens}.")

        # Ensure model's config reflects the final state of lm_head vocab size
        if self.config.vocab_size != self.lm_head.out_features:
             logger.warning(f"Config vocab_size ({self.config.vocab_size}) mismatch with lm_head.out_features ({self.lm_head.out_features}) after resize. Correcting config.")
             self.config.vocab_size = self.lm_head.out_features

        return self.get_output_embeddings()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Any]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[ # Return signature updated
        Optional[torch.Tensor], # loss
        torch.Tensor,           # logits
        Optional[Tuple[torch.Tensor, ...]], # hidden_states from model
        Optional[Tuple[Optional[torch.Tensor], ...]], # attentions from model
        Optional[List[Any]],    # past_key_values from model
        Optional[torch.Tensor], # total_lb_loss from model
        Optional[torch.Tensor]  # total_rz_loss from model
    ]:
        
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        hidden_states_from_model = model_outputs[0]
        # model_outputs also contains: all_hs_out, all_att_out, all_kv_cache_out, total_lb_loss, total_rz_loss
        # These are model_outputs[1] to model_outputs[5]
        
        text_logits_hidden_states = hidden_states_from_model
        if self.config.multimodal and pixel_values is not None and past_key_values is None: # Prefill stage with image
            if input_ids is not None: # Ensure text tokens are present to slice for
                L_text_original = input_ids.shape[1]
                # Image tokens are prepended, so text hidden states are at the end
                text_start_idx = hidden_states_from_model.shape[1] - L_text_original
                if text_start_idx >= 0:
                     text_logits_hidden_states = hidden_states_from_model[:, text_start_idx:, :]
                else: # Should not happen if logic is correct
                    logger.error("Error slicing text hidden states for multimodal logits. text_start_idx < 0.")
            # If only image is passed (input_ids is None), then text_logits_hidden_states remains full hidden_states_from_model
            # which might be an issue if no text is expected for logits. This case should be handled by user or training script.
            
        logits = self.lm_head(text_logits_hidden_states)
        
        loss = None
        if labels is not None:
            # Standard cross-entropy loss for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            current_ignore_index = -100 
            # Heuristic: if -100 is not used anywhere in labels, assume pad_token_id should be ignored.
            if not (labels == -100).any():
                current_ignore_index = self.config.pad_token_id

            # Ensure sequence lengths match for loss calculation after shifting
            if shift_logits.shape[1] != shift_labels.shape[1]:
                # This can happen if, e.g., logits correspond to full sequence (img+text) but labels only to text part.
                # Or if there's a mismatch in how labels are prepared.
                # A common strategy: labels should align with the part of logits used for loss.
                # If text_logits_hidden_states was correctly sliced, then shift_logits and shift_labels should align.
                # However, if labels are shorter (e.g. only for text part, and logits include image part not used for loss),
                # then this slicing needs to be robust or labels need to be padded.

                # For now, assume labels are for the same part as logits used (text_logits_hidden_states).
                # If a mismatch still occurs, take the minimum sequence length.
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                if min_len == 0: # Avoid error if sequence becomes empty
                    logger.warning("Logits or labels have zero sequence length after shifting and potential slicing. Main loss will be 0 or error.")
                    # Setting main_loss to a tensor to allow addition of aux losses
                    main_loss = torch.tensor(0.0, device=logits.device, requires_grad=self.training) # Ensure grad if training
                else:
                    shift_logits = shift_logits[:, :min_len, :]
                    shift_labels = shift_labels[:, :min_len]
                    loss_fct = nn.CrossEntropyLoss(ignore_index=current_ignore_index)
                    main_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            elif shift_logits.shape[1] == 0: # Sequence length was 1, so after shifting it's 0
                 logger.warning("Sequence length is 1, so shift_logits/labels are empty. Main loss set to 0.")
                 main_loss = torch.tensor(0.0, device=logits.device, requires_grad=self.training)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=current_ignore_index)
                main_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            loss = main_loss
            # Add auxiliary losses from the model if they exist (i.e., not None)
            total_lb_loss = model_outputs[4] # total_lb_loss from ApertisModel
            total_rz_loss = model_outputs[5] # total_rz_loss from ApertisModel

            if total_lb_loss is not None:
                loss += total_lb_loss
            if total_rz_loss is not None:
                loss += total_rz_loss

        # The return tuple from ApertisModel is:
        # (hidden_s, all_hs_out, all_att_out, all_kv_cache_out, final_lb_loss, final_rz_loss)
        # So model_outputs[0] is hidden_s
        # model_outputs[1] is all_hs_out
        # model_outputs[2] is all_att_out
        # model_outputs[3] is all_kv_cache_out
        # model_outputs[4] is final_lb_loss
        # model_outputs[5] is final_rz_loss
        # We need to return: (loss, logits, all_hs_out, all_att_out, all_kv_cache_out, final_lb_loss, final_rz_loss)

        return (loss, logits) + model_outputs[1:] # This correctly appends all items from model_outputs[1] onwards
    
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, past_key_values: Optional[List[Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids_override: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        
        B, L_curr = input_ids.shape
        
        current_input_ids_for_model = input_ids
        pos_ids_for_model = None

        if past_key_values is not None:
            current_input_ids_for_model = input_ids[:, -1:]
            
            past_kv_len = 0
            if self.config.attention_type == "standard_mha" and past_key_values[0] is not None:
                past_kv_len = past_key_values[0][0].shape[1]
            elif self.config.attention_type == "selective_ssm" and past_key_values[0] is not None:
                past_kv_len = past_key_values[0][0].shape[2]

            current_absolute_position = attention_mask.shape[1] - 1
            pos_ids_for_model = torch.tensor([[current_absolute_position]], dtype=torch.long, device=input_ids.device).expand(B, -1)

        else:
            if position_ids_override is not None:
                pos_ids_for_model = position_ids_override
            else:
                pos_ids_for_model = torch.arange(L_curr, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(B, -1)
        
        if position_ids_override is not None:
            pos_ids_for_model = position_ids_override

        model_inputs = {
            "input_ids": current_input_ids_for_model,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": pos_ids_for_model,
            "use_cache": kwargs.get("use_cache", True),
        }

        if past_key_values is None and "pixel_values" in kwargs:
            model_inputs["pixel_values"] = kwargs["pixel_values"]
        
        return model_inputs
    
    @torch.no_grad()
    def generate(
        self, input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        max_new_tokens: Optional[int] = 20,
        min_new_tokens: Optional[int] = 0,
        do_sample: Optional[bool] = False,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        use_cache: bool = True, **kwargs,
    ) -> torch.Tensor:
        if input_ids is None: raise ValueError("input_ids must be provided.")
        batch_size, prompt_len = input_ids.shape
        
        temp = max(temperature, 1e-6) if do_sample else 1.0
        
        eos_ids_final = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        if eos_ids_final is None: eos_ids_final = [] 
        if not isinstance(eos_ids_final, list): eos_ids_final = [eos_ids_final]
        
        current_pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        if current_pad_token_id is None:
            logger.warning("pad_token_id is None, generation might behave unexpectedly with padding.")
            current_pad_token_id = 0

        current_attention_mask = attention_mask
        if current_attention_mask is None:
            current_attention_mask = torch.ones_like(input_ids)

        prompt_position_ids = position_ids
        if prompt_position_ids is None:
            prompt_position_ids = torch.arange(prompt_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        num_effective_prompt_tokens = prompt_len
        current_pixel_values = pixel_values
        
        if self.config.multimodal and current_pixel_values is not None:
            dummy_img_config = self.config
            num_img_tokens_est = (dummy_img_config.image_size // dummy_img_config.vision_patch_size)**2 +1
            
            img_padding_mask = torch.ones((batch_size, num_img_tokens_est), dtype=current_attention_mask.dtype, device=current_attention_mask.device)
            current_attention_mask = torch.cat([img_padding_mask, current_attention_mask], dim=1)
            
            img_pos_ids = torch.arange(num_img_tokens_est, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
            text_pos_ids_shifted = prompt_position_ids + num_img_tokens_est
            prompt_position_ids = torch.cat([img_pos_ids, text_pos_ids_shifted], dim=1)
            num_effective_prompt_tokens += num_img_tokens_est

        generated_tokens = input_ids
        internal_past_kv = None
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        
        for step in range(max_new_tokens):
            input_ids_for_prep = generated_tokens if internal_past_kv is None else generated_tokens[:, -1:]
            position_ids_for_prep = prompt_position_ids if internal_past_kv is None else None

            model_inputs = self.prepare_inputs_for_generation(
                input_ids=input_ids_for_prep,
                past_key_values=internal_past_kv,
                attention_mask=current_attention_mask,
                position_ids_override=position_ids_for_prep,
                pixel_values=current_pixel_values,
                use_cache=use_cache
            )
            
            if current_pixel_values is not None: current_pixel_values = None
            
            outputs = self(
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=model_inputs["position_ids"],
                past_key_values=model_inputs["past_key_values"],
                pixel_values=model_inputs.get("pixel_values"),
                use_cache=model_inputs["use_cache"],
            )
            
            next_token_logits = outputs[1][:, -1, :]
            internal_past_kv = outputs[4] if use_cache else None
            
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    if unfinished_sequences[i] == 0: continue
                    for token_id_in_seq in generated_tokens[i]:
                        if token_id_in_seq < next_token_logits.shape[-1]:
                             next_token_logits[i, token_id_in_seq] /= repetition_penalty
            
            if do_sample:
                if temp != 1.0: next_token_logits = next_token_logits / temp
                if top_k > 0:
                    top_k_vals, _ = torch.topk(next_token_logits, top_k)
                    kth_val = top_k_vals[:, -1].unsqueeze(-1)
                    next_token_logits.masked_fill_(next_token_logits < kth_val, float("-inf"))
                if top_p < 1.0:
                    s_logits, s_indices = torch.sort(next_token_logits, descending=True)
                    cum_probs = torch.cumsum(F.softmax(s_logits, dim=-1), dim=-1)
                    s_indices_to_remove = cum_probs > top_p
                    s_indices_to_remove[..., 1:] = s_indices_to_remove[..., :-1].clone()
                    s_indices_to_remove[..., 0] = 0
                    indices_to_remove = torch.zeros_like(next_token_logits,dtype=torch.bool).scatter_(-1,s_indices,s_indices_to_remove)
                    next_token_logits.masked_fill_(indices_to_remove, float("-inf"))
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            next_tokens = next_tokens * unfinished_sequences + current_pad_token_id * (1 - unfinished_sequences)
            generated_tokens = torch.cat([generated_tokens, next_tokens.unsqueeze(-1)], dim=-1)
            
            current_attention_mask = torch.cat(
                [current_attention_mask, unfinished_sequences.unsqueeze(-1)], dim=1
            )
            
            for eos_id_val in eos_ids_final:
                if eos_id_val is not None:
                    unfinished_sequences = unfinished_sequences.masked_fill((next_tokens == eos_id_val) & (unfinished_sequences == 1), 0)
            
            if unfinished_sequences.max() == 0 and (generated_tokens.shape[1] - prompt_len) >= min_new_tokens :
                break
        return generated_tokens

    def gradient_checkpointing_enable(self):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

def create_apertis_model(
    model_size: str = "base",
    vocab_size_override: Optional[int] = None,
    multimodal: bool = False,
    use_flash_attention: bool = False,
    use_expert_system: bool = False,
    attention_type_override: Optional[str] = None,
    ssm_d_inner: Optional[int] = None,
    ssm_d_state: int = 16,
    ssm_dt_rank: Union[int, str] = "auto",
    ssm_conv_kernel: int = 4,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ApertisForCausalLM:
    model_configs_presets = {
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
    }
    if model_size not in model_configs_presets:
        raise ValueError(f"Unsupported model_size: {model_size}. Choose from {list(model_configs_presets.keys())}")
    
    final_config_dict = model_configs_presets[model_size].copy()
    
    if vocab_size_override is not None: 
        final_config_dict["vocab_size"] = vocab_size_override
    
    if attention_type_override is not None:
        final_config_dict["attention_type"] = attention_type_override
    else:
        final_config_dict.setdefault("attention_type", "standard_mha")

    final_config_dict.update({
        "multimodal": multimodal, 
        "use_flash_attention": use_flash_attention,
        "use_expert_system": use_expert_system,
        "ssm_d_inner": ssm_d_inner,
        "ssm_d_state": ssm_d_state,
        "ssm_dt_rank": ssm_dt_rank,
        "ssm_conv_kernel": ssm_conv_kernel,
    })

    if config_overrides:
        final_config_dict.update(config_overrides)

    if final_config_dict["hidden_size"] % final_config_dict["num_attention_heads"] != 0:
        logger.warning(f"hidden_size {final_config_dict['hidden_size']} not divisible by num_attention_heads {final_config_dict['num_attention_heads']}. Adjusting num_attention_heads.")
        for i in range(final_config_dict["num_attention_heads"], 0, -1):
            if final_config_dict["hidden_size"] % i == 0:
                final_config_dict["num_attention_heads"] = i
                break
        if final_config_dict["hidden_size"] % final_config_dict["num_attention_heads"] != 0:
             final_config_dict["num_attention_heads"] = 1

    config = ApertisConfig(**final_config_dict)
    model = ApertisForCausalLM(config)
    return model

# --- New functions for parameter-based scaling ---

def parse_param_count(param_str: Union[str, int]) -> int:
    """
    Parses a parameter count string (e.g., "10M", "1.5B") into an integer.
    Also accepts integers directly.
    """
    if isinstance(param_str, int):
        if param_str < 1_000_000: # Assuming numbers less than 1M are direct counts
             logger.warning(f"Interpreting direct integer {param_str} as raw parameter count. If you meant millions or billions, use M or B suffix.")
        return param_str

    s = str(param_str).strip().upper()
    if not s:
        raise ValueError("Parameter string cannot be empty.")

    multiplier = 1
    if s.endswith('K'):
        multiplier = 1_000
        s = s[:-1]
    elif s.endswith('M'):
        multiplier = 1_000_000
        s = s[:-1]
    elif s.endswith('B'):
        multiplier = 1_000_000_000
        s = s[:-1]

    try:
        val = float(s)
    except ValueError:
        raise ValueError(f"Invalid numeric value in parameter string: '{param_str}'")

    return int(val * multiplier)

def _calculate_params_for_dims(
    vocab_size: int, hidden_size: int, num_layers: int, intermediate_size: int,
    tie_word_embeddings: bool = True, use_expert_system: bool = False, num_experts: int = 0
) -> int:
    """Helper to estimate parameters for given dimensions."""
    params = 0
    # Embeddings
    params += vocab_size * hidden_size
    if not tie_word_embeddings: # LM head
        params += vocab_size * hidden_size

    # Attention blocks (Q,K,V,O projections per layer)
    params += num_layers * (4 * hidden_size * hidden_size)

    # FFN blocks
    if use_expert_system and num_experts > 0:
        # Each expert has a standard FFN structure
        ffn_params_per_expert = 2 * hidden_size * intermediate_size # (up_proj + down_proj)
        params += num_layers * num_experts * ffn_params_per_expert
        # Router parameters (simplified: one linear layer per FFN block)
        params += num_layers * (hidden_size * num_experts)
    else:
        params += num_layers * (2 * hidden_size * intermediate_size) # (up_proj + down_proj)

    # LayerNorms (approx, simplified: 2 per layer: attn pre-norm, ffn pre-norm) + final norm
    # Each LayerNorm has 2*hidden_size params (weight + bias)
    params += (2 * num_layers + 1) * (2 * hidden_size)

    return params

def calculate_model_dimensions(
    target_params_str: Union[str, int],
    vocab_size: int,
    use_expert_system: bool = False,
    num_experts_target: int = 8, # Desired number of experts if MoE
    # experts_per_token_target: int = 2, # Not directly used in param count, but good for config
    min_hidden_size: int = 256,
    max_hidden_size: int = 8192, # Max practical hidden size to search
    min_layers: int = 2,
    max_layers: int = 128, # Max practical layers
    head_dim_preference: int = 64,
    intermediate_multiple_of: int = 256, # For FFN intermediate size, often multiple of 256
    intermediate_ratio: float = 4.0, # Standard ratio for FFN intermediate size
    tie_word_embeddings: bool = True,
) -> Dict[str, Any]:
    """
    Calculates model dimensions (hidden_size, num_hidden_layers, num_attention_heads, intermediate_size)
    to approximately match a target parameter count.
    Prioritizes finding a suitable hidden_size and num_layers.
    """
    target_params = parse_param_count(target_params_str)
    if not (10_000_000 <= target_params <= 70_000_000_000):
        logger.warning(f"Target parameters {target_params_str} ({target_params}) is outside the typical 10M-70B range. Results may be suboptimal.")

    best_config = None
    min_diff = float('inf')

    # Heuristic: Start with a reasonable hidden_size and adjust layers, then vice-versa
    # This is a simplified search. More sophisticated methods (e.g., geometric progression for h, L) could be used.

    # Iterate through potential number of layers
    for l in range(min_layers, max_layers + 1, 2): # Iterate layers by steps of 2 for smoother scaling
        # Estimate hidden_size needed for this many layers
        # Simplified formula: P_core = 12*L*h^2 for non-MoE, P_core_moe = L * (4h^2 + E * 8h^2) for MoE
        # P_approx = P_core + 2*V*h (embeddings)
        # For non-MoE: target_params - 2*V*h_est ~ 12*L*h^2 => h ~ sqrt((target_params - 2*V*h_est) / (12*L))
        # For MoE: target_params - 2*V*h_est ~ L*(4*h^2 + E*intermediate_ratio*h*hidden_size)
        # This becomes complex to solve directly for h. Iterative approach is better.

        # Iterate through potential hidden sizes (multiples of head_dim_preference or some step)
        current_h = min_hidden_size
        while current_h <= max_hidden_size:
            h = current_h

            # Ensure h is multiple of some sensible value, e.g., 64 or head_dim_preference
            # And also ensure num_attention_heads can be derived cleanly
            if h % head_dim_preference != 0:
                h = ((h // head_dim_preference) + 1) * head_dim_preference
            if h == 0 : h = head_dim_preference # Avoid h=0
            if h > max_hidden_size: break


            num_attention_heads = h // head_dim_preference
            if num_attention_heads == 0: num_attention_heads = 1 # Ensure at least 1 head
            if h % num_attention_heads != 0: # Should not happen if h is multiple of head_dim_preference
                h = num_attention_heads * head_dim_preference # Adjust h to be divisible

            intermediate_size_raw = int(h * intermediate_ratio)
            # Make intermediate_size a multiple of `intermediate_multiple_of`
            intermediate_size = ((intermediate_size_raw + intermediate_multiple_of -1) // intermediate_multiple_of) * intermediate_multiple_of
            if intermediate_size == 0 : intermediate_size = intermediate_multiple_of


            current_params = _calculate_params_for_dims(
                vocab_size, h, l, intermediate_size,
                tie_word_embeddings, use_expert_system, num_experts_target if use_expert_system else 0
            )

            diff = abs(current_params - target_params)

            if diff < min_diff:
                min_diff = diff
                best_config = {
                    "hidden_size": h,
                    "num_hidden_layers": l,
                    "num_attention_heads": num_attention_heads,
                    "intermediate_size": intermediate_size,
                    "calculated_params": current_params,
                    "target_params": target_params,
                    "param_diff": diff
                }

            # Heuristic to step hidden_size:
            # If current_params < target_params, we need to increase h.
            # If current_params > target_params, we might have overshot for this L, try next L or break.
            if current_params > target_params and diff > min_diff: # If we're over and getting worse
                 break # Stop increasing h for this L, as it will only get further

            # Step hidden_size. Make step larger for larger h.
            step_h = max(head_dim_preference, h // 16)
            current_h += step_h
            if current_h > max_hidden_size and best_config is None: # Ensure we try at least one max_hidden_size
                current_h = max_hidden_size


    if not best_config:
        # Fallback: if no config found (e.g., target_params too small/large for constraints)
        # Try to generate a small default config
        logger.warning(f"Could not find a good configuration for {target_params_str}. Using a fallback small config.")
        h = min_hidden_size
        l_fallback = min_layers
        num_attention_heads_fallback = max(1, h // head_dim_preference)
        intermediate_size_fallback = int(h * intermediate_ratio)
        intermediate_size_fallback = ((intermediate_size_fallback + intermediate_multiple_of -1) // intermediate_multiple_of) * intermediate_multiple_of
        calculated_fallback_params = _calculate_params_for_dims(
            vocab_size, h, l_fallback, intermediate_size_fallback,
            tie_word_embeddings, use_expert_system, num_experts_target if use_expert_system else 0
        )
        return {
            "hidden_size": h,
            "num_hidden_layers": l_fallback,
            "num_attention_heads": num_attention_heads_fallback,
            "intermediate_size": intermediate_size_fallback,
            "calculated_params": calculated_fallback_params,
            "target_params": target_params,
            "param_diff": abs(calculated_fallback_params - target_params),
            "備考": "Fallback configuration"
        }

    logger.info(f"Calculated dimensions for target ~{target_params/1e6:.2f}M params: "
                f"H={best_config['hidden_size']}, L={best_config['num_hidden_layers']}, A={best_config['num_attention_heads']}, I={best_config['intermediate_size']}. "
                f"Resulted in {best_config['calculated_params']/1e6:.2f}M params (Diff: {best_config['param_diff']}).")
    return best_config

def estimate_model_parameters(config: ApertisConfig) -> int:
    """
    Estimates the total number of parameters in an Apertis model given its configuration.
    """
    # Embedding parameters
    embedding_params = config.vocab_size * config.hidden_size

    # LM head parameters (if not tied)
    lm_head_params = 0
    if not config.tie_word_embeddings:
        lm_head_params = config.vocab_size * config.hidden_size

    # Transformer layer parameters
    layer_params = 0
    # Attention sublayer
    # Q, K, V projections: 3 * hidden_size * hidden_size
    # Output projection: hidden_size * hidden_size
    attention_params_per_layer = 4 * config.hidden_size * config.hidden_size

    # Feed-forward sublayer
    ffn_params_per_layer = 0
    if config.use_expert_system and config.num_experts > 0:
        # Each expert has its own FFN weights
        ffn_params_one_expert = 2 * config.hidden_size * config.intermediate_size # up_proj + down_proj
        ffn_params_per_layer = config.num_experts * ffn_params_one_expert
        # Add router parameters: hidden_size * num_experts for the linear layer in router
        ffn_params_per_layer += config.hidden_size * config.num_experts
    else:
        ffn_params_per_layer = 2 * config.hidden_size * config.intermediate_size # up_proj + down_proj

    layer_params = config.num_hidden_layers * (attention_params_per_layer + ffn_params_per_layer)

    # LayerNorm parameters (approximate)
    # Each LayerNorm has 2 * hidden_size parameters (weight and bias)
    # Typically 2 LayerNorms per transformer layer (before attention, before FFN)
    # Plus one final LayerNorm after all layers.
    layernorm_params = (2 * config.num_hidden_layers + 1) * (2 * config.hidden_size)

    # Absolute position embeddings (if used)
    abs_pos_params = 0
    if config.position_embedding_type == "absolute":
        abs_pos_params = config.max_position_embeddings * config.hidden_size

    # Multimodal parameters (simplified, as UnifiedMultimodalEncoder can be complex)
    # This is a rough estimate, true calculation would need to inspect multimodal_encoder internals.
    multimodal_params = 0
    if config.multimodal:
        # Vision projection layer
        if config.vision_embed_dim != config.hidden_size:
            multimodal_params += config.vision_embed_dim * config.hidden_size # Linear projection
        # A rough estimate for a small vision tower or adapter.
        # This is highly dependent on the actual vision_layers, vision_heads etc.
        # For simplicity, let's add a placeholder amount or assume it's part of vision_embed_dim complexity.
        # A more accurate count would require instantiating UnifiedMultimodalEncoder or knowing its formula.
        # Placeholder: e.g., 50M for a typical ViT adapter.
        # For now, this part is an underestimation if a large vision model is part of the encoder.
        # The current UnifiedMultimodalEncoder in the provided code doesn't seem to have its own large backbone.
        # It seems to imply pixel_values are already embeddings from a vision model.
        # If UnifiedMultimodalEncoder itself contains a ViT:
        # Roughly: vision_layers * (4 * vision_embed_dim^2 (attn) + 2 * vision_embed_dim * (4*vision_embed_dim) (FFN))
        # This part is complex and depends on the specifics of UnifiedMultimodalEncoder.
        # The current `UnifiedMultimodalEncoder` is not defined in `core.py`, so we assume it's a simple projector or adapter.
        pass


    total_params = (
        embedding_params + lm_head_params + layer_params + layernorm_params +
        abs_pos_params + multimodal_params
    )

    return total_params

# --- End of new functions ---

def create_apertis_model(
    target_param_count: Union[str, int] = "125M", # Changed from model_size
    vocab_size_override: Optional[int] = None,
    multimodal: bool = False,
    use_flash_attention: bool = False,
    use_expert_system: bool = False,
    num_experts_target_override: Optional[int] = None, # Allow overriding calculated/default num_experts
    experts_per_token_target_override: Optional[int] = None,
    attention_type_override: Optional[str] = None,
    ssm_d_inner: Optional[int] = None,
    ssm_d_state: int = 16,
    ssm_dt_rank: Union[int, str] = "auto",
    ssm_conv_kernel: int = 4,
    config_overrides: Optional[Dict[str, Any]] = None,
) -> ApertisForCausalLM:
    # model_configs_presets = { # This is now replaced by calculation
    #     "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
    #     "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
    #     "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
    # }
    # if model_size not in model_configs_presets: # Remove this check
    #     raise ValueError(f"Unsupported model_size: {model_size}. Choose from {list(model_configs_presets.keys())}")

    # final_config_dict = model_configs_presets[model_size].copy() # Replace this

    # Determine vocab_size: override > config_overrides > default
    # The default ApertisConfig vocab_size is 32000.
    # We need a vocab_size to pass to calculate_model_dimensions.
    # If config_overrides has vocab_size, use that. Otherwise, use default.
    # vocab_size_override will be applied *after* calculation if provided.

    temp_config_for_vocab = ApertisConfig(**(config_overrides or {}))
    vocab_size_for_calculation = temp_config_for_vocab.vocab_size
    if vocab_size_override is not None: # If override is present, it's the final truth for calculation too.
        vocab_size_for_calculation = vocab_size_override

    calculated_dims = calculate_model_dimensions(
        target_params_str=target_param_count,
        vocab_size=vocab_size_for_calculation, # Use a sensible default or allow override
        use_expert_system=use_expert_system,
        num_experts_target=num_experts_target_override if num_experts_target_override is not None else 8, # Default if MoE
    )

    final_config_dict = {
        "hidden_size": calculated_dims["hidden_size"],
        "num_hidden_layers": calculated_dims["num_hidden_layers"],
        "num_attention_heads": calculated_dims["num_attention_heads"],
        "intermediate_size": calculated_dims["intermediate_size"],
    }
    logger.info(f"Target parameters: {target_param_count}. Calculated config: {final_config_dict}")
    logger.info(f"Smart calculator estimated: {calculated_dims.get('calculated_params', 'N/A')} params "
                f"(diff: {calculated_dims.get('param_diff', 'N/A')})")

    if vocab_size_override is not None:
        final_config_dict["vocab_size"] = vocab_size_override
    elif "vocab_size" not in final_config_dict: # Ensure vocab_size is set
        final_config_dict["vocab_size"] = vocab_size_for_calculation

    if attention_type_override is not None:
        final_config_dict["attention_type"] = attention_type_override
    else:
        final_config_dict.setdefault("attention_type", "standard_mha")

    final_config_dict.update({
        "multimodal": multimodal,
        "use_flash_attention": use_flash_attention, # This will be passed to ApertisConfig
        "use_expert_system": use_expert_system,
        # If use_expert_system is true, num_experts and experts_per_token should be set.
        # ApertisConfig has defaults (8, 2). We can override them here if needed.
        "ssm_d_inner": ssm_d_inner,
        "ssm_d_state": ssm_d_state,
        "ssm_dt_rank": ssm_dt_rank,
        "ssm_conv_kernel": ssm_conv_kernel,
    })

    if use_expert_system:
        if num_experts_target_override is not None:
            final_config_dict["num_experts"] = num_experts_target_override
        else: # use default from ApertisConfig or what was used in calculation
            final_config_dict.setdefault("num_experts", calculated_dims.get("num_experts", 8))

        if experts_per_token_target_override is not None:
            final_config_dict["experts_per_token"] = experts_per_token_target_override
        else:
            final_config_dict.setdefault("experts_per_token", calculated_dims.get("experts_per_token", 2))


    if config_overrides:
        # Config_overrides should take precedence over calculated dimensions if there's a conflict,
        # but calculated dimensions form the base.
        base_calculated_config = final_config_dict.copy()
        base_calculated_config.update(config_overrides) # User overrides can trump calculated ones
        final_config_dict = base_calculated_config

    # Ensure hidden_size is divisible by num_attention_heads after all overrides
    if final_config_dict["hidden_size"] % final_config_dict["num_attention_heads"] != 0:
        logger.warning(
            f"Final hidden_size {final_config_dict['hidden_size']} not divisible by "
            f"num_attention_heads {final_config_dict['num_attention_heads']}. "
            f"Adjusting num_attention_heads for divisibility by trying factors or setting to 1."
        )
        h_final = final_config_dict["hidden_size"]
        current_heads = final_config_dict["num_attention_heads"]

        # Try to maintain head_dim if possible, by adjusting heads to make h_final divisible
        preferred_head_dim = h_final // current_heads if current_heads > 0 else 64 # Estimate original target head_dim
        if preferred_head_dim == 0: preferred_head_dim = 64

        if h_final % preferred_head_dim == 0 and h_final // preferred_head_dim > 0 :
            final_config_dict["num_attention_heads"] = h_final // preferred_head_dim
        else: # Fallback: find largest factor or set to 1
            found_factor = False
            for i in range(min(current_heads, h_final), 0, -1):
                if h_final % i == 0:
                    final_config_dict["num_attention_heads"] = i
                    found_factor = True
                    break
            if not found_factor:
                 final_config_dict["num_attention_heads"] = 1
        logger.info(f"Adjusted num_attention_heads to {final_config_dict['num_attention_heads']}")


    config = ApertisConfig(**final_config_dict)

    # Log the actual parameters of the configured model
    actual_params = estimate_model_parameters(config)
    logger.info(f"Model configured with H={config.hidden_size}, L={config.num_hidden_layers}, A={config.num_attention_heads}, I={config.intermediate_size}, V={config.vocab_size}.")
    logger.info(f"Estimated actual parameters for this configuration: {actual_params/1e6:.2f}M. Target was ~{parse_param_count(target_param_count)/1e6:.2f}M.")

    model = ApertisForCausalLM(config)
    return model