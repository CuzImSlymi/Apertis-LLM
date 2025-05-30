import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import numpy as np
import sys
import os
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

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

    @classmethod
    def from_dict(cls, config_dict):
        import inspect
        sig = inspect.signature(cls.__init__)
        valid_keys = {param.name for param in sig.parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD}
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        if "ssm_dt_rank" in filtered_config_dict and filtered_config_dict["ssm_dt_rank"] == "auto":
            hs = filtered_config_dict.get("hidden_size", 768)
            filtered_config_dict["ssm_dt_rank"] = math.ceil(hs / 16)
        return cls(**filtered_config_dict)
    
    def to_dict(self):
        return self.__dict__
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        import json
        config_file_path_str = ""
        if os.path.isdir(model_name_or_path):
            config_file_path_str = os.path.join(model_name_or_path, "config.json")
        elif os.path.isfile(model_name_or_path) and model_name_or_path.endswith(".json"):
            config_file_path_str = model_name_or_path
        else:
            config_file_path_str = os.path.join(model_name_or_path, "config.json")
        if not os.path.exists(config_file_path_str) and os.path.isdir(model_name_or_path):
            parent_dir_config_file = os.path.join(Path(model_name_or_path).parent, "config.json")
            if os.path.exists(parent_dir_config_file):
                config_file_path_str = parent_dir_config_file
        if not os.path.exists(config_file_path_str):
            raise FileNotFoundError(f"Config file not found for '{model_name_or_path}'. Looked for: '{config_file_path_str}'")
        with open(config_file_path_str, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def save_pretrained(self, save_directory: str):
        import json
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
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 8, experts_per_token: int = 2, activation_function: str = "gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = min(num_experts, experts_per_token)
        self.router_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, intermediate_size),
                self._get_activation_fn(activation_function),
                nn.Dropout(0.1),
                nn.Linear(intermediate_size, hidden_size)
            ) for _ in range(num_experts)
        ])
        
    def _get_activation_fn(self, act_fn_str: str):
        if act_fn_str == "gelu": return nn.GELU()
        elif act_fn_str == "relu": return nn.ReLU()
        elif act_fn_str == "silu" or act_fn_str == "swish": return nn.SiLU()
        raise ValueError(f"Unsupported activation: {act_fn_str}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, L, D = hidden_states.shape
        normalized_hidden = self.router_norm(hidden_states)
        router_logits = self.router(normalized_hidden).float()
        routing_weights, selected_indices = torch.topk(router_logits, self.experts_per_token, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1).to(hidden_states.dtype)
        final_output = torch.zeros_like(hidden_states)
        for k_idx in range(self.experts_per_token):
            expert_indices_for_slot_k = selected_indices[:, :, k_idx]
            weights_for_slot_k = routing_weights[:, :, k_idx].unsqueeze(-1)
            for expert_actual_idx in range(self.num_experts):
                mask = (expert_indices_for_slot_k == expert_actual_idx)
                if mask.any():
                    selected_hidden_states = hidden_states[mask]
                    if selected_hidden_states.shape[0] > 0:
                        expert_out = self.experts[expert_actual_idx](selected_hidden_states)
                        current_expert_weights = weights_for_slot_k[mask]
                        final_output[mask] += current_expert_weights * expert_out
        return final_output

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

            if use_c and past_kv is not None:
                key_states = torch.cat([past_kv[0], key_states], dim=1)
                value_states = torch.cat([past_kv[1], value_states], dim=1)
            
            present_cache = (key_states, value_states) if use_c else None
            
            query_layer = self._transpose_for_scores(query_states)
            key_layer = self._transpose_for_scores(key_states)
            value_layer = self._transpose_for_scores(value_states)

            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)

            if att_mask is not None:
                attention_scores = attention_scores + att_mask
            
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            att_weights_proxy = attention_probs if output_att else None
            
            attention_probs = self.attention_dropout(attention_probs)
            
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
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
        if config.use_expert_system:
            self.ffn = AdaptiveExpertSystem(
                config.hidden_size, config.intermediate_size, config.num_experts,
                config.experts_per_token, config.hidden_act
            )
        else:
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
    
    def forward(self, hidden_s: torch.Tensor) -> torch.Tensor:
        residual = hidden_s
        normed_hidden_s = self.pre_norm(hidden_s)
        ffn_out = self.ffn(normed_hidden_s)
        dropped_ffn_out = self.output_dropout(ffn_out)
        return dropped_ffn_out + residual

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        
        x_for_attn = hidden_s
        if self.rope and self.config.attention_type == "standard_mha":
             x_for_attn = self.rope(hidden_s, position_ids=pos_ids)

        att_out, att_w, present_att_cache = self.attention(
            x_for_attn, att_mask, pos_ids, past_kv, output_att, use_c
        )
        layer_out = self.feed_forward(att_out)
        return layer_out, att_w, present_att_cache

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
    
    def resize_token_embeddings(self, new_num_tokens: int):
        old_embeddings = self.token_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, self.config.hidden_size, device=old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        self._init_weights(new_embeddings) # Initialize new embeddings

        # Copy old weights
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        self.token_embeddings = new_embeddings
        self.config.vocab_size = new_num_tokens
        self.vocab_size = new_num_tokens
        # Update padding_idx if it's out of bounds for the new vocab size, though it usually isn't.
        if self.padding_idx is not None and self.padding_idx >= new_num_tokens:
            logger.warning(f"Padding idx {self.padding_idx} is out of new vocab size {new_num_tokens}. Setting to 0.")
            self.padding_idx = 0 # Or handle as error/configurable
            self.token_embeddings.padding_idx = self.padding_idx
        logger.info(f"Resized token embeddings to {new_num_tokens} tokens.")
        return self.token_embeddings

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        combined_attention_mask = None
        if input_shape[-1] > 1:
            L_q = input_shape[1]
            L_kv = past_key_values_length + L_q
            
            full_causal_mask = torch.ones((L_kv, L_kv), dtype=torch.bool, device=inputs_embeds.device).tril(diagonal=0)
            causal_mask_for_current_q = full_causal_mask[past_key_values_length:, :]
            causal_mask_for_current_q = causal_mask_for_current_q[None, None, :, :].expand(
                input_shape[0],1, L_q, L_kv
            )

            if attention_mask is not None: # attention_mask is (B, L_kv_total)
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
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]], Optional[Tuple[Optional[torch.Tensor], ...]], Optional[List[Any]]]:
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
            if self.config.attention_type == "standard_mha":
                past_kv_len = past_key_values[0][0].shape[1]
            elif self.config.attention_type == "selective_ssm":
                past_kv_len = past_key_values[0][0].shape[2]
        
        current_pos_ids = position_ids
        if current_pos_ids is None:
            current_pos_ids = torch.arange(past_kv_len, L_curr_query + past_kv_len, dtype=torch.long, device=inputs_embeds.device)
            current_pos_ids = current_pos_ids.unsqueeze(0).expand(B, -1)

        if self.config.position_embedding_type == "absolute" and self.abs_pos_embeddings:
            abs_pos_embeds = self.abs_pos_embeddings(current_pos_ids)
            inputs_embeds = inputs_embeds + abs_pos_embeds
        
        query_embeddings_for_layers = inputs_embeds
        pos_ids_for_layers = current_pos_ids
        
        num_img_tokens = 0
        if self.config.multimodal and pixel_values is not None and past_kv_len == 0:
            img_feats = self.multimodal_encoder(pixel_values)
            img_feats_proj = self.vision_projection(img_feats)
            num_img_tokens = img_feats_proj.shape[1]

            query_embeddings_for_layers = torch.cat([img_feats_proj, inputs_embeds], dim=1)
            
            img_pos_ids = torch.arange(num_img_tokens, dtype=torch.long, device=inputs_embeds.device).unsqueeze(0).expand(B, -1)
            text_pos_ids_shifted = current_pos_ids + num_img_tokens
            pos_ids_for_layers = torch.cat([img_pos_ids, text_pos_ids_shifted], dim=1)
            
            if attention_mask is not None and attention_mask.shape[1] == L_curr_query:
                img_padding_mask = torch.ones((B, num_img_tokens), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([img_padding_mask, attention_mask], dim=1)
            elif attention_mask is None:
                attention_mask = torch.ones((B, num_img_tokens + L_curr_query), dtype=torch.long, device=query_embeddings_for_layers.device)

        elif self.config.multimodal and pixel_values is not None and past_kv_len > 0:
            logger.warning("pixel_values provided with past_key_values. Ignoring image for this step.")

        hidden_s = self.embed_dropout(query_embeddings_for_layers)
        
        ext_att_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (B, query_embeddings_for_layers.shape[1]),
            query_embeddings_for_layers,
            past_kv_len
        )
        
        all_hs_out, all_att_out, all_kv_cache_out = [], [], []
        for i, layer_mod in enumerate(self.layers):
            if output_hs: all_hs_out.append(hidden_s)
            
            layer_past_kv = past_key_values[i] if past_key_values and i < len(past_key_values) else None
            current_layer_pos_ids = pos_ids_for_layers
            
            if self.gradient_checkpointing and self.training and not use_c:
                def create_cp_forward(mod):
                    def _cp_forward(h_states, att_m_layer, p_ids_layer, pkvs_layer):
                        return mod(h_states, att_m_layer, p_ids_layer, pkvs_layer, output_att, use_c)[0]
                    return _cp_forward
                current_layer_cp_forward = create_cp_forward(layer_mod)
                hidden_s = torch.utils.checkpoint.checkpoint(
                    current_layer_cp_forward, hidden_s, ext_att_mask, current_layer_pos_ids, layer_past_kv,
                    use_reentrant=False
                )
                if output_att: all_att_out.append(None)
                if use_c: all_kv_cache_out.append(None)
            else:
                hidden_s, att_w, present_kv = layer_mod(
                    hidden_s, ext_att_mask, current_layer_pos_ids, layer_past_kv, output_att, use_c
                )
                if output_att: all_att_out.append(att_w)
                if use_c: all_kv_cache_out.append(present_kv)
        
        hidden_s = self.final_post_norm(hidden_s)
        if output_hs: all_hs_out.append(hidden_s)
        
        return (
            hidden_s,
            tuple(all_hs_out) if output_hs and all_hs_out else None,
            tuple(all_att_out) if output_att and all_att_out else None,
            tuple(all_kv_cache_out) if use_c and all_kv_cache_out else None
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

    def resize_token_embeddings(self, new_num_tokens: int):
        self.model.resize_token_embeddings(new_num_tokens)
        # If lm_head is not tied or needs separate resizing
        if not self.config.tie_word_embeddings or self.lm_head.weight is not self.model.token_embeddings.weight:
            old_lm_head = self.lm_head
            self.lm_head = nn.Linear(self.config.hidden_size, new_num_tokens, bias=False, device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype)
            self._init_weights(self.lm_head) # Initialize new lm_head weights
            
            num_tokens_to_copy = min(old_lm_head.out_features, new_num_tokens)
            self.lm_head.weight.data[:num_tokens_to_copy, :] = old_lm_head.weight.data[:num_tokens_to_copy, :]
        elif self.config.tie_word_embeddings:
             self.lm_head.weight = self.model.token_embeddings.weight # Re-tie
             # Update out_features for the tied head (nn.Linear doesn't auto-update this if weight is replaced)
             self.lm_head.out_features = new_num_tokens


        self.config.vocab_size = new_num_tokens # Update config as well
        logger.info(f"Resized LM head to {new_num_tokens} tokens.")
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
    ) -> Tuple[
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[Tuple[torch.Tensor, ...]],
        Optional[Tuple[Optional[torch.Tensor], ...]],
        Optional[List[Any]]
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
        
        text_logits_hidden_states = hidden_states_from_model
        if self.config.multimodal and pixel_values is not None and past_key_values is None:
            if input_ids is not None:
                L_text_original = input_ids.shape[1]
                text_start_index = hidden_states_from_model.shape[1] - L_text_original
                if text_start_index >= 0:
                     text_logits_hidden_states = hidden_states_from_model[:, text_start_index:, :]
                else:
                    logger.error("Error slicing text hidden states for multimodal logits.")
            
        logits = self.lm_head(text_logits_hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            current_ignore_index = -100 # Default for fine-tuning where labels include -100
            # If labels do not contain -100 (e.g. pre-training), use pad_token_id for ignoring padding.
            # This assumes labels are never -100 during pre-training.
            if not (labels == -100).any():
                current_ignore_index = self.config.pad_token_id

            if shift_logits.shape[1] != shift_labels.shape[1]:
                min_len = min(shift_logits.shape[1], shift_labels.shape[1])
                shift_logits = shift_logits[:, :min_len, :]
                shift_labels = shift_labels[:, :min_len]
                if min_len == 0:
                    logger.warning("Logits or labels have zero sequence length after shifting. Loss will be 0 or error.")
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=current_ignore_index)
                    loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            elif shift_logits.shape[1] == 0:
                 logger.warning("Sequence length is 1, so shift_logits/labels are empty. Loss set to 0.")
                 loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss_fct = nn.CrossEntropyLoss(ignore_index=current_ignore_index)
                loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
        return (loss, logits) + model_outputs[1:]
    
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
) -> ApertisForCausalLM:
    model_configs_presets = {
        "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
        "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
        "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
    }
    if model_size not in model_configs_presets:
        raise ValueError(f"Unsupported model_size: {model_size}. Choose from {list(model_configs_presets.keys())}")
    config_dict = model_configs_presets[model_size].copy()
    if vocab_size_override is not None: config_dict["vocab_size"] = vocab_size_override
    
    if attention_type_override is not None:
        config_dict["attention_type"] = attention_type_override
    else:
        config_dict.setdefault("attention_type", "standard_mha")


    config_dict.update({
        "multimodal": multimodal, "use_flash_attention": use_flash_attention,
        "use_expert_system": use_expert_system,
        "ssm_d_inner": ssm_d_inner,
        "ssm_d_state": ssm_d_state,
        "ssm_dt_rank": ssm_dt_rank,
        "ssm_conv_kernel": ssm_conv_kernel,
    })
    if config_dict["hidden_size"] % config_dict["num_attention_heads"] != 0:
        logger.warning(f"hidden_size {config_dict['hidden_size']} not divisible by num_attention_heads {config_dict['num_attention_heads']}. Adjusting num_attention_heads.")
        for i in range(config_dict["num_attention_heads"], 0, -1):
            if config_dict["hidden_size"] % i == 0:
                config_dict["num_attention_heads"] = i
                break
        if config_dict["hidden_size"] % config_dict["num_attention_heads"] != 0:
             config_dict["num_attention_heads"] = 1

    config = ApertisConfig(**config_dict)
    model = ApertisForCausalLM(config)
    return model