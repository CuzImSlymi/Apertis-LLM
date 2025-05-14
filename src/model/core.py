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

# Ensure the multimodal module can be imported if this file is run directly for tests.
# This assumes 'src' is the parent of 'model' and 'multimodal'.
if __name__ == '__main__' and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.multimodal.module import UnifiedMultimodalEncoder


class ApertisConfig:
    """Core configuration for the Apertis architecture."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12, # H
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
        attention_type: str = "selective_ssm", 
        # SSM specific parameters (inspired by Mamba)
        ssm_d_inner: Optional[int] = None, # If None, will be H * N for SSM, or 2*D for FFN-like expansion
        ssm_d_state: int = 16,       # N, state dimension per head
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
        self.intermediate_size = intermediate_size # For FFN in MoE or standard FFN
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

        # SSM specific parameters
        self.ssm_d_state = ssm_d_state # N
        # If attention is SSM, d_inner is H * N by Mamba-like construction.
        # User can still provide ssm_d_inner for non-SSM attention types or future flexibility.
        if self.attention_type == "selective_ssm":
            # For headed SSMs, d_inner becomes the sum of head_states or an expansion based on it.
            # In our Mamba-like setup, the main pathway (x_conv, z) uses d_inner.
            # The SSM scan output per head is d_state. Concatenated H*d_state.
            # For gating y_ssm * silu(z), these two need to match.
            # So, d_inner for projections (in_proj_x, in_proj_z) MUST be H * d_state.
            derived_ssm_d_inner = self.num_attention_heads * self.ssm_d_state
            if ssm_d_inner is not None and ssm_d_inner != derived_ssm_d_inner:
                logger.warning(f"For attention_type='selective_ssm', ssm_d_inner ({ssm_d_inner}) "
                               f"is overridden by num_attention_heads*ssm_d_state ({derived_ssm_d_inner}).")
            self.ssm_d_inner = derived_ssm_d_inner
        elif ssm_d_inner is None: # For other attention types or if user doesn't specify
            self.ssm_d_inner = 2 * self.hidden_size # Default expansion for non-SSM or general use
        else: # User provided ssm_d_inner for a non-SSM attention type
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
        # Remove keys from config_dict that are not in __init__ signature
        # This prevents errors if old config files with extra keys are loaded.
        filtered_config_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        # Ensure ssm_dt_rank is int if it was string "auto" in old config and not re-calculated
        if "ssm_dt_rank" in filtered_config_dict and filtered_config_dict["ssm_dt_rank"] == "auto":
            # Need hidden_size to calculate, assume it's in filtered_config_dict
            hs = filtered_config_dict.get("hidden_size", 768) # Default if not found
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
        else: # Try assuming model_name_or_path is a dir even if .json not specified
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
        self.register_buffer("cos_cached", cos_cached.unsqueeze(0).unsqueeze(2), persistent=False) 
        self.register_buffer("sin_cached", sin_cached.unsqueeze(0).unsqueeze(2), persistent=False) 


    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[-2] 

        cos = self.cos_cached[:, :seq_len, ...] 
        sin = self.sin_cached[:, :seq_len, ...] 

        if x.ndim == 3: 
            cos = cos.squeeze(2) 
            sin = sin.squeeze(2) 
        
        x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2) 
        x1 = x_reshaped[..., 0] 
        x2 = x_reshaped[..., 1] 
        
        rotated_x_part1 = x1 * cos - x2 * sin
        rotated_x_part2 = x1 * sin + x2 * cos
        
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = rotated_x_part1
        x_out[..., 1::2] = rotated_x_part2
        
        return x_out.type_as(x)


class SelectiveLinearAttention(nn.Module): 
    def __init__(self, config: ApertisConfig): 
        super().__init__()

        self.hidden_size = config.hidden_size      
        self.num_heads = config.num_attention_heads 
        
        # For SSM, d_inner is effectively H * N (state_dim per head)
        # This ensures consistency for gating with z.
        self.d_inner = self.num_heads * config.ssm_d_state
        self.d_state = config.ssm_d_state # N per head         
        self.dt_rank = config.ssm_dt_rank
        self.conv_kernel_size = config.ssm_conv_kernel
        
        # head_d_inner is the dimension of u_scan input per head, which is d_state (N)
        self.head_d_inner = self.d_state # Each head processes a state of size N
                                         # Input u_scan will be projected to this.
                                         # This might require adjustment to in_proj_x/z or u_scan input.
                                         # Mamba: in_proj makes d_inner. Conv keeps d_inner.
                                         # x_activated (d_inner) is input to SSM projections and silu(z)
                                         # If d_inner is H*N, then u_scan can be (B,H,L,N)
        
        # Projections from D to d_inner (which is H*N)
        self.in_proj_x = nn.Linear(self.hidden_size, self.d_inner, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.d_inner, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=self.conv_kernel_size, groups=self.d_inner, # Depthwise
            padding=(self.conv_kernel_size - 1) # Causal padding
        )

        # x_param_proj: input d_inner (from x_activated), output dt_rank + H*N (for B) + H*N (for C)
        self.x_param_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.num_heads * self.d_state, bias=False)
        
        self.dt_proj_head = nn.Linear(self.dt_rank, self.num_heads, bias=True)
        torch.nn.init.uniform_(self.dt_proj_head.bias, a=math.log(1e-3), b=math.log(1e-2))

        self.A_log = nn.Parameter(torch.empty(self.num_heads, self.d_state))
        # Initialize A_log to small negative values, so exp(A_log) is close to 1 but < 1 for stability
        torch.nn.init.uniform_(self.A_log, a=math.log(0.1), b=math.log(0.9))


        self.D = nn.Parameter(torch.ones(self.d_inner)) # Skip connection, applied to x_activated

        # Output projection from d_inner (H*N) back to D
        self.out_proj = nn.Linear(self.d_inner, self.hidden_size, bias=False)
        
        self.use_cache = False 
        self.conv_state = None
        self.ssm_state = None

    def _ssm_pytorch_scan(self, u_scan_input, delta, A_log_resolved, B_params, C_params, ssm_state_prev=None):
        # u_scan_input: (B, H, L, N=d_state), this is x_ssm in Mamba paper Fig 3.
        # delta: (B, H, L, 1)
        # A_log_resolved: (H, N=d_state)
        # B_params: (B, H, L, N=d_state), this is data-dependent B_k for recurrence h_k = A_bar_k h_{k-1} + B_bar_k u_k
        # C_params: (B, H, L, N=d_state), this is data-dependent C_k for y_k = C_k h_k
        
        B, H, L, N = u_scan_input.shape # N here is d_state (ssm_d_state)

        A_cont_diag = torch.exp(A_log_resolved).neg() # (H, N)
        
        # delta_A for A_bar = exp(delta * A)
        delta_A = delta * A_cont_diag.unsqueeze(0).unsqueeze(2) # (B,H,L,1) * (1,H,1,N) -> (B,H,L,N)
        A_bar = torch.exp(delta_A) 

        # B_bar_k * u_k term.
        # In Mamba, B is projected from x, then discretized with delta & A.
        # And u is also projected from x.
        # Here, B_params is already the data-dependent B for the recurrence.
        # u_scan_input is the u_k.
        # So the term to add is B_params * u_scan_input (element-wise if B_params is structured for it)
        # Or if B_params is a matrix to multiply u_scan_input.
        # Given B_params is (B,H,L,N) and u_scan_input is (B,H,L,N), assume B_params already incorporates delta and input.
        # This B_params is B_ssm from forward(). It's the (B_bar * x) term.
        
        # Let B_effective = delta * B_params * u_scan_input # If B_params was like continuous B from projection
        # But forward already made B_ssm = B_params.view(...).transpose. It's (B,H,L,N).
        # This B_ssm IS the B_k term (incorporating B_bar_k * u_k) to be added to recurrent state.

        if self.use_cache and ssm_state_prev is not None:
            h = ssm_state_prev 
        else:
            h = torch.zeros(B, H, N, device=u_scan_input.device, dtype=u_scan_input.dtype)

        ys = []
        for i in range(L):
            h = A_bar[:,:,i,:] * h + B_params[:,:,i,:] # B_params is the B_k * u_k term
            # y_i = C_k * h_k
            # C_params is (B,H,L,N). h is (B,H,N). y_i should be (B,H,N)
            ys.append(C_params[:,:,i,:] * h) 

        y_stacked = torch.stack(ys, dim=2) # (B, H, L, N)
        
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

        # 1. Input projections (x and z) to d_inner = H*N
        x_projected = self.in_proj_x(hidden_states) # (B, L, H*N)
        z_gate_features = self.in_proj_z(hidden_states) # (B, L, H*N)

        # 2. 1D Convolution on x_projected
        x_conv_input = x_projected.transpose(1, 2) # (B, H*N, L) for Conv1D
        if conv_state_prev is not None and use_cache:
            if conv_state_prev.shape[1] == self.d_inner and conv_state_prev.shape[2] == self.conv_kernel_size -1 : 
                 x_conv_input = torch.cat([conv_state_prev, x_conv_input], dim=2)
            else:
                 logger.warning(f"Mismatched conv_state shape: {conv_state_prev.shape}, vs x_conv_input: {x_conv_input.shape} expected K-1={self.conv_kernel_size-1}")
        
        current_conv_state = x_conv_input[:, :, -(self.conv_kernel_size - 1):].detach() if use_cache else None
        
        x_convolved = self.conv1d(x_conv_input)[:, :, :L] # Ensure output length is L (causal padding handles this)
        x_convolved = x_convolved.transpose(1, 2) # (B, L, H*N)
        
        # x_activated is the input to parameter projections for dt,B,C and also main input u to SSM
        x_activated = F.silu(x_convolved) # (B, L, H*N)

        # 3. Project for SSM params (dt, B, C) from x_activated
        ssm_params_raw = self.x_param_proj(x_activated) # (B, L, dt_rank + 2*H*N)
        
        dt_rank_feats, B_raw_params, C_raw_params = torch.split(
            ssm_params_raw, 
            [self.dt_rank, self.num_heads * self.d_state, self.num_heads * self.d_state], 
            dim=-1
        ) 
        
        # Delta: dt_rank_feats (B,L,dt_rank) -> dt_logits (B,L,H) -> delta (B,H,L,1)
        delta_logits = self.dt_proj_head(dt_rank_feats) 
        delta = F.softplus(delta_logits).transpose(1,2).unsqueeze(-1) 

        # B_ssm_term and C_ssm_term for the scan
        # B_raw_params is (B,L, H*N). Reshape to (B,L,H,N). Transpose to (B,H,L,N)
        B_ssm_term = B_raw_params.view(B, L, self.num_heads, self.d_state).transpose(1,2)
        C_ssm_term = C_raw_params.view(B, L, self.num_heads, self.d_state).transpose(1,2)
        
        # A_log is (H,N)

        # Input u to scan: x_activated (B,L,H*N) needs to be (B,H,L,N) if each head processes N-dim state from its slice of H*N
        # This means u_scan should be a view of x_activated where each head gets N features.
        # head_dim_for_scan_input = self.d_state (N)
        # d_inner = H * N was established.
        u_scan_input = x_activated.view(B, L, self.num_heads, self.d_state).transpose(1,2) # (B,H,L,N)
        
        scan_output_tuple = self._ssm_pytorch_scan(u_scan_input, delta, self.A_log, B_ssm_term, C_ssm_term, ssm_state_prev)
        
        y_ssm_scan_output, current_ssm_h_state = None, None
        if use_cache:
            y_ssm_scan_output, current_ssm_h_state = scan_output_tuple
        else:
            y_ssm_scan_output = scan_output_tuple # (B,H,L,N_state)

        # y_ssm_scan_output is (B,H,L,N_state). Reshape to (B,L, H*N_state = d_inner)
        y_ssm_processed = y_ssm_scan_output.transpose(1,2).contiguous().view(B, L, self.d_inner)
        
        # Add D * x_activated term (Mamba-style skip before final gating)
        y_ssm_plus_skip = y_ssm_processed + self.D.unsqueeze(0).unsqueeze(0) * x_activated
        
        # Gate with z_gate_features (B,L,d_inner)
        output_gated = y_ssm_plus_skip * F.silu(z_gate_features)
        
        # Final output projection
        final_output = self.out_proj(output_gated) # (B, L, D_hidden)

        current_cache_state = None
        if use_cache:
            current_cache_state = (current_conv_state, current_ssm_h_state)
        
        # For output_attentions, return y_ssm_processed (B,L,H*N) as a proxy for "context"
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
        if config.attention_type == "selective_ssm" or config.attention_type == "selective_linear": 
            if config.attention_type == "selective_linear":
                 logger.warning("Config: 'selective_linear' aliased to 'selective_ssm'.")
            self.attention_mechanism = SelectiveLinearAttention(config=config) 
        else:
            logger.warning(f"Config: Using standard nn.MultiheadAttention for attention_type='{config.attention_type}'.")
            self.attention_mechanism = nn.MultiheadAttention(
                config.hidden_size, config.num_attention_heads, 
                dropout=config.attention_probs_dropout_prob, batch_first=True
            )
            self.mha_o_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.pre_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) 
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_s: torch.Tensor, att_mask: Optional[torch.Tensor]=None, 
                pos_ids: Optional[torch.Tensor]=None, past_kv: Optional[Any]=None, 
                output_att: bool=False, use_c: bool=False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        
        residual = hidden_s
        normed_hidden_s = self.pre_norm(hidden_s)
        
        att_out, att_weights_proxy, present_cache = None, None, None

        if isinstance(self.attention_mechanism, SelectiveLinearAttention): 
            att_out, att_weights_proxy, present_cache = self.attention_mechanism(
                normed_hidden_s, attention_mask=att_mask, 
                position_ids=pos_ids, past_key_value=past_kv, 
                output_attentions=output_att, use_cache=use_c
            )
        elif isinstance(self.attention_mechanism, nn.MultiheadAttention):
            key_padding_mask_bool = None
            if att_mask is not None: 
                if att_mask.ndim == 4:
                    key_padding_mask_bool = (att_mask.squeeze(1).squeeze(1) < -1e4) 
                elif att_mask.ndim == 2: 
                    key_padding_mask_bool = (att_mask == 0)

            att_out, att_weights_proxy = self.attention_mechanism(
                normed_hidden_s, normed_hidden_s, normed_hidden_s, 
                key_padding_mask=key_padding_mask_bool, 
                need_weights=output_att
            )
            att_out = self.mha_o_proj(att_out)
        else:
            raise TypeError(f"Unknown attention mechanism type: {type(self.attention_mechanism)}")

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
        if self.rope: 
             x_for_attn = self.rope(hidden_s) 


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
        
        if input_ids is not None and inputs_embeds is not None: raise ValueError("Specify one of input_ids or inputs_embeds.")
        
        if inputs_embeds is None:
            if input_ids is None: raise ValueError("input_ids required if inputs_embeds is None.")
            inputs_embeds = self.token_embeddings(input_ids) 
        
        B, L_txt, D_ = inputs_embeds.shape
        
        past_len = 0
        if past_key_values is not None and past_key_values[0] is not None: 
            if attention_mask is not None and L_txt < attention_mask.shape[1]: 
                past_len = attention_mask.shape[1] - L_txt

        current_pos_ids = position_ids 
        if current_pos_ids is None: 
            current_pos_ids = torch.arange(past_len, L_txt + past_len, dtype=torch.long, device=inputs_embeds.device)
            current_pos_ids = current_pos_ids.unsqueeze(0).expand(B, -1)


        if self.config.position_embedding_type == "absolute" and self.abs_pos_embeddings:
            abs_pos_embeds = self.abs_pos_embeddings(current_pos_ids)
            inputs_embeds = inputs_embeds + abs_pos_embeds
        
        final_embeds = inputs_embeds
        final_att_mask = attention_mask 
        final_pos_ids_for_layers = current_pos_ids 

        if self.multimodal_encoder and pixel_values is not None:
            if past_len == 0: 
                img_feats = self.multimodal_encoder(pixel_values) 
                img_feats_proj = self.vision_projection(img_feats) 
                L_img = img_feats_proj.shape[1]
                
                final_embeds = torch.cat([img_feats_proj, inputs_embeds], dim=1)
                
                if final_att_mask is not None: 
                    img_att_mask = torch.ones((B, L_img), dtype=final_att_mask.dtype, device=final_att_mask.device)
                    final_att_mask = torch.cat([img_att_mask, final_att_mask], dim=1) 
                
                img_pos_ids_for_layers = torch.arange(L_img, dtype=torch.long, device=final_embeds.device).unsqueeze(0).expand(B, -1)
                text_pos_ids_shifted_for_layers = current_pos_ids + L_img 
                final_pos_ids_for_layers = torch.cat([img_pos_ids_for_layers, text_pos_ids_shifted_for_layers], dim=1)
            
        hidden_s = self.embed_dropout(final_embeds)
        
        ext_att_mask = None
        if final_att_mask is not None:
            ext_att_mask = final_att_mask.unsqueeze(1).unsqueeze(2)
            ext_att_mask = ext_att_mask.to(dtype=hidden_s.dtype)
            ext_att_mask = (1.0 - ext_att_mask) * torch.finfo(hidden_s.dtype).min

        all_hs_out, all_att_out, all_kv_cache_out = [], [], []
        
        for i, layer_mod in enumerate(self.layers):
            if output_hs: all_hs_out.append(hidden_s)
            layer_past_kv = past_key_values[i] if past_key_values and i < len(past_key_values) else None 
            
            
            if self.gradient_checkpointing and self.training and not use_c: 
                def create_cp_forward(mod): 
                    def _cp_forward(h_states, att_m, p_ids, pkvs): 
                        return mod(h_states, att_m, p_ids, pkvs, output_attentions, use_cache)[0] 
                    return _cp_forward
                
                current_layer_cp_forward = create_cp_forward(layer_mod)
                hidden_s = torch.utils.checkpoint.checkpoint(
                    current_layer_cp_forward, hidden_s, ext_att_mask, final_pos_ids_for_layers, layer_past_kv, 
                    use_reentrant=False 
                )
                if output_att: all_att_out.append(None) 
                if use_c: all_kv_cache_out.append(None) 
            else:
                hidden_s, att_w, present_kv = layer_mod(
                    hidden_s, ext_att_mask, final_pos_ids_for_layers, layer_past_kv, output_att, use_c
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
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
            past_key_values=past_key_values, inputs_embeds=inputs_embeds, pixel_values=pixel_values,
            use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states,
        )
        
        last_hidden_state = model_outputs[0]
        logits = self.lm_head(last_hidden_state)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id) 
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return (loss, logits) + model_outputs[1:]
    
    def prepare_inputs_for_generation(
        self, input_ids: torch.Tensor, past_key_values: Optional[List[Any]] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
        **kwargs 
    ) -> Dict[str, Any]:
        
        current_input_ids = input_ids
        current_position_ids = None 

        current_seq_len = input_ids.shape[1] 
        batch_size = input_ids.shape[0]

        if past_key_values is not None:
            current_input_ids = input_ids[:, -1].unsqueeze(-1)
            if attention_mask is not None:
                current_token_position = attention_mask.shape[1] - 1
                current_position_ids = torch.tensor([[current_token_position]], dtype=torch.long, device=input_ids.device).expand(batch_size, -1)
        else:
            if kwargs.get("position_ids") is not None:
                 current_position_ids = kwargs["position_ids"]
            elif current_seq_len > 0 : 
                 current_position_ids = torch.arange(current_seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)


        model_inputs = {
            "input_ids": current_input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask, 
            "position_ids": current_position_ids, 
            "use_cache": kwargs.get("use_cache", True),
        }
        
        if past_key_values is None and "pixel_values" in kwargs: 
            model_inputs["pixel_values"] = kwargs["pixel_values"]
        
        return model_inputs
    
    @torch.no_grad()
    def generate(
        self, input_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None, 
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
        if not isinstance(eos_ids_final, list): eos_ids_final = [eos_ids_final]
        
        current_pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id

        current_attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids)
        
        generated_tokens = input_ids
        internal_past_kv = None
        
        current_pixel_values = pixel_values

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)

        for _ in range(max_new_tokens):
            
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_tokens, 
                past_key_values=internal_past_kv,
                attention_mask=current_attention_mask, 
                pixel_values=current_pixel_values, 
                use_cache=use_cache,
            )
            if current_pixel_values is not None: current_pixel_values = None 

            outputs = self.model( 
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                position_ids=model_inputs["position_ids"],
                past_key_values=model_inputs["past_key_values"],
                pixel_values=model_inputs.get("pixel_values"), 
                use_cache=model_inputs["use_cache"],
            )
            
            last_hidden_state = outputs[0] 
            internal_past_kv = outputs[3]  

            next_token_logits = self.lm_head(last_hidden_state[:, -1, :]) 

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
            current_attention_mask = torch.cat([current_attention_mask, unfinished_sequences.unsqueeze(-1)], dim=1) 
            
            for eos_id_val in eos_ids_final: # CORRECTED HERE
                unfinished_sequences = unfinished_sequences.masked_fill((next_tokens == eos_id_val) & (unfinished_sequences == 1), 0) 
            
            if unfinished_sequences.max() == 0 and generated_tokens.shape[1] - prompt_len >= min_new_tokens:
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
    
    config_dict.update({
        "multimodal": multimodal, "use_flash_attention": use_flash_attention,
        "use_expert_system": use_expert_system, "attention_type": "selective_ssm",
        "ssm_d_inner": ssm_d_inner, # Will be resolved in ApertisConfig based on attention_type
        "ssm_d_state": ssm_d_state,
        "ssm_dt_rank": ssm_dt_rank, # Will be resolved in ApertisConfig
        "ssm_conv_kernel": ssm_conv_kernel,
    })
    
    config = ApertisConfig(**config_dict)
    model = ApertisForCausalLM(config)
    return model
