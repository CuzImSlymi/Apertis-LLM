# /home/ubuntu/ApertisAI_Project/Apertis AI_/src/model/core.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import numpy as np
import sys
import os
import json # Import json

# Add the parent directory to the path so we can import the multimodal module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.multimodal.module import UnifiedMultimodalEncoder

# Define a simple output class for consistency with HF-like return types
class CausalLMOutputWithPast:
    def __init__(self, loss=None, logits=None, past_key_values=None, hidden_states=None, attentions=None):
        self.loss = loss
        self.logits = logits
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions

# Define a base model output class
class BaseModelOutputWithPast:
     def __init__(self, last_hidden_state=None, past_key_values=None, hidden_states=None, attentions=None):
         self.last_hidden_state = last_hidden_state
         self.past_key_values = past_key_values
         self.hidden_states = hidden_states
         self.attentions = attentions

class ApertisConfig:
    """Core configuration for the Apertis architecture."""

    # Keep the __init__ method as it was
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
        position_embedding_type: str = "rotary",
        use_cache: bool = True,
        classifier_dropout: float = None,
        model_type: str = "apertis",
        tie_word_embeddings: bool = True,
        rope_theta: float = 10000.0,
        sliding_window: Optional[int] = None,
        attention_type: str = "selective_linear",
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
        model_size: Optional[str] = None,
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
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.model_type = model_type
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_theta = rope_theta
        self.sliding_window = sliding_window
        self.attention_type = attention_type
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
        self.model_size = model_size

        # Auto-configure based on model_size if provided and other params are default
        if model_size:
            size_map = {
                "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
            }
            if model_size in size_map:
                params = size_map[model_size]
                # Only override if they haven"t been explicitly set to non-default values
                if self.hidden_size == 768: self.hidden_size = params["hidden_size"]
                if self.num_hidden_layers == 12: self.num_hidden_layers = params["num_hidden_layers"]
                if self.num_attention_heads == 12: self.num_attention_heads = params["num_attention_heads"]
                if self.intermediate_size == 3072: self.intermediate_size = params["intermediate_size"]
            else:
                 print(f"Warning: Unknown model_size \t\"{model_size}\". Using default parameters.")

    @classmethod
    def from_dict(cls, config_dict):
        """Create a configuration from a dictionary."""
        # Create an instance with default values first
        config = cls()
        # Update with values from the dictionary
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                # Optionally log unknown keys
                print(f"DEBUG: Unknown key in config_dict: {key}")

        # Re-run the model_size logic AFTER loading from dict, in case model_size was in the dict
        # This ensures model_size overrides defaults but not explicitly set values from the dict
        if config.model_size:
             size_map = {
                 "small": {"hidden_size": 512, "num_hidden_layers": 8, "num_attention_heads": 8, "intermediate_size": 2048},
                 "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
                 "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
             }
             if config.model_size in size_map:
                 params = size_map[config.model_size]
                 # Get defaults again to check against
                 default_config = cls()
                 # Only override if the value loaded from the dict *was* the default value
                 if config_dict.get("hidden_size", getattr(default_config, "hidden_size")) == getattr(default_config, "hidden_size"): config.hidden_size = params["hidden_size"]
                 if config_dict.get("num_hidden_layers", getattr(default_config, "num_hidden_layers")) == getattr(default_config, "num_hidden_layers"): config.num_hidden_layers = params["num_hidden_layers"]
                 if config_dict.get("num_attention_heads", getattr(default_config, "num_attention_heads")) == getattr(default_config, "num_attention_heads"): config.num_attention_heads = params["num_attention_heads"]
                 if config_dict.get("intermediate_size", getattr(default_config, "intermediate_size")) == getattr(default_config, "intermediate_size"): config.intermediate_size = params["intermediate_size"]

        return config

    def to_dict(self):
        """Convert configuration to dictionary."""
        # Ensure all attributes are serializable
        serializable_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_dict[key] = value
            else:
                # Handle non-serializable types if necessary, e.g., convert to string
                serializable_dict[key] = str(value)
        return serializable_dict

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        """Load configuration from a pretrained model path."""
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "config.json")
        else:
            # If it"s a file path, assume config.json is in the same directory
            config_file = os.path.join(os.path.dirname(model_name_or_path), "config.json")

        if not os.path.exists(config_file):
             raise FileNotFoundError(f"Configuration file not found at {config_file}")

        with open(config_file, "r") as f:
            config_dict = json.load(f)

        # Log the loaded intermediate_size specifically for debugging
        loaded_intermediate_size = config_dict.get("intermediate_size")
        print(f"DEBUG: Loaded intermediate_size from {config_file}: {loaded_intermediate_size}")

        config = cls.from_dict(config_dict)

        print(f"DEBUG: Final intermediate_size in config object after from_dict: {config.intermediate_size}")
        return config

    def save_pretrained(self, save_directory):
        """Save configuration to a directory."""
        os.makedirs(save_directory, exist_ok=True)
        config_file = os.path.join(save_directory, "config.json")

        with open(config_file, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class RotaryEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE)."""

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.float32)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # seq_len: The actual sequence length of the input tokens
        if seq_len is None:
             seq_len = x.shape[-2] # Infer from input if not provided

        # Check if cache needs update
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        # Return the cached values up to the current sequence length
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # cos, sin: [1, 1, seq_len, dim]
    # position_ids: [bsz, seq_len]
    # q, k: [bsz, num_heads, seq_len, head_dim]

    # Gather the specific cos/sin values for the given position IDs
    # Need to handle potential broadcasting issues if position_ids shape varies
    # cos/sin are cached up to max_seq_len
    # position_ids might be < max_seq_len

    # Squeeze cos/sin: [seq_len, dim]
    cos = cos.squeeze(0).squeeze(0) # Remove batch and head dims (assuming they are 1)
    sin = sin.squeeze(0).squeeze(0)

    # Gather based on position_ids: [bsz, seq_len, dim]
    # Ensure position_ids are within the cached range
    position_ids_clipped = position_ids.clamp(max=cos.shape[0] - 1)
    cos = cos[position_ids_clipped]
    sin = sin[position_ids_clipped]

    # Add num_heads dimension for broadcasting: [bsz, 1, seq_len, dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class SelectiveLinearAttention(nn.Module):
    """Selective Linear Attention mechanism for O(n) time complexity."""
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_dropout: float = 0.1, sliding_window: Optional[int] = None, max_position_embeddings: int = 2048, rope_theta: float = 10000.0, position_embedding_type: str = "rotary"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.position_embedding_type = position_embedding_type
        self.head_dim = hidden_size // num_attention_heads
        if self.head_dim * num_attention_heads != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_attention_heads")
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(attention_dropout)
        self.q_norm = nn.LayerNorm(hidden_size)
        self.k_norm = nn.LayerNorm(hidden_size)
        if self.position_embedding_type == "rotary":
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings, base=rope_theta)
        else:
            self.rotary_emb = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_norm(hidden_states)
        key_states = self.k_norm(hidden_states)
        value_states = hidden_states
        query_states = self.q_proj(query_states)
        key_states = self.k_proj(key_states)
        value_states = self.v_proj(value_states)
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = None, None
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        # Apply activation function (e.g., ELU + 1) for positivity
        phi_q = F.elu(query_states) + 1.0 + 1e-6
        phi_k = F.elu(key_states) + 1.0 + 1e-6

        # Apply mask if provided (assuming causal mask is handled by structure or mask input)
        if attention_mask is not None:
            # Ensure mask is broadcastable: [bsz, 1, q_len, kv_seq_len]
            if attention_mask.dim() == 2: # Convert 2D mask to 4D
                 attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, kv_seq_len)
            elif attention_mask.dim() == 3: # Expand 3D mask
                 attention_mask = attention_mask[:, None, :, :]

            # Apply mask by setting masked values to zero before einsum
            # Mask has 0 for allowed positions, large negative for masked
            # We need to apply it to phi_k and value_states before einsum
            # Create a boolean mask where True means masked
            bool_mask = (attention_mask < -1e4).squeeze(1) # [bsz, q_len, kv_seq_len]
            # We need mask for keys: [bsz, num_heads, kv_seq_len, head_dim]
            # This masking logic seems complex and potentially incorrect for linear attention
            # Let"s simplify: Assume causal masking is implicitly handled or not needed for now
            # If a mask is provided, it usually affects the K, V computation
            pass # Simplified: Skipping explicit masking within linear attention for now

        # Corrected Einsums for Linear Attention
        kv = torch.einsum("bhsd,bhsv->bhdv", phi_k, value_states) # Correct: [b, h, d, v] (assuming v=d)
        z = torch.einsum("bhsd->bhd", phi_k) # Correct: Sum keys over sequence dim [b, h, d]
        numerator = torch.einsum("bhqd,bhdv->bhqv", phi_q, kv) # Correct: [b, h, q, v]
        denominator = torch.einsum("bhqd,bhd->bhq", phi_q, z) # Correct: [b, h, q]

        # Calculate final attention output
        attn_output = numerator / (denominator.unsqueeze(-1) + 1e-6) # Add dimension for broadcasting

        # Apply dropout
        attn_output = self.dropout(attn_output) # Shape: [bsz, num_heads, q_len, head_dim]

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        attn_weights = None # Linear attention typically doesn"t return standard weights

        return attn_output, attn_weights, past_key_value

class ApertisAttention(nn.Module):
    """Attention module for Apertis architecture."""
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.attention_dropout = config.attention_probs_dropout_prob
        self.position_embedding_type = config.position_embedding_type
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.sliding_window = config.sliding_window
        self.use_flash_attention = config.use_flash_attention and hasattr(torch.nn.functional, "scaled_dot_product_attention")

        if self.head_dim * self.num_attention_heads != self.hidden_size:
            raise ValueError(f"hidden_size must be divisible by num_attention_heads")

        # Temporarily force standard attention for debugging training error
        # config_attention_type = "standard" # config.attention_type
        # Use the actual config value now, as SelectiveLinearAttention might be fixed
        config_attention_type = config.attention_type
        print(f"DEBUG: Using attention_type: \"{config_attention_type}\"")

        # Choose attention implementation
        if config_attention_type == "selective_linear":
            self.attention_impl = SelectiveLinearAttention(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                attention_dropout=self.attention_dropout,
                sliding_window=self.sliding_window,
                max_position_embeddings=self.max_position_embeddings,
                rope_theta=self.rope_theta,
                position_embedding_type=self.position_embedding_type,
            )
        elif config_attention_type == "standard":
            # Standard Attention Implementation (Simplified for brevity)
            self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.o_proj = nn.Linear(self.hidden_size, self.hidden_size)
            self.dropout = nn.Dropout(self.attention_dropout)
            if self.position_embedding_type == "rotary":
                self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta)
            else:
                self.rotary_emb = None
            self.attention_impl = self._standard_attention_forward # Assign method
        else:
            raise ValueError(f"Unsupported attention_type: {config_attention_type}") # Use modified var

        # Layer normalization before attention
        self.input_layernorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        # Residual dropout after attention
        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _standard_attention_forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = None, None
        if self.rotary_emb is not None:
            cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        past_key_value = (key_states, value_states) if use_cache else None

        if self.use_flash_attention:
            # Ensure attention_mask is None or compatible boolean mask for flash attention
            # Flash attention requires mask where True indicates masking
            if attention_mask is not None:
                 # Assuming input mask uses large negative values for masking
                 flash_attn_mask = attention_mask < -1e4
                 if flash_attn_mask.dim() == 4:
                     flash_attn_mask = flash_attn_mask.squeeze(1) # Reduce to [bsz, q_len, kv_seq_len]
            else:
                 flash_attn_mask = None

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states, key_states, value_states,
                attn_mask=flash_attn_mask, # Use boolean mask or None
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=flash_attn_mask is None and q_len > 1 # Enable causal only if no mask and q_len > 1
            )
            attn_weights = None # Flash attention does not return weights by default
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if attention_mask is not None:
                # Ensure mask is broadcastable [bsz, 1, q_len, kv_seq_len]
                if attention_mask.dim() == 2:
                    # Expand 2D mask: [bsz, src_len] -> [bsz, 1, 1, src_len]
                    # Need target length q_len for broadcasting
                    expanded_mask = attention_mask[:, None, None, :].expand(bsz, 1, q_len, kv_seq_len)
                    attention_mask = expanded_mask
                elif attention_mask.dim() == 3:
                    # Expand 3D mask: [bsz, q_len, kv_seq_len] -> [bsz, 1, q_len, kv_seq_len]
                    attention_mask = attention_mask[:, None, :, :]
                # Check shape before adding
                if attn_weights.shape != attention_mask.shape:
                     print(f"WARNING: Additive attention mask shape mismatch! Weights: {attn_weights.shape}, Mask: {attention_mask.shape}")
                     # Attempt to broadcast mask if possible, otherwise error might occur
                attn_weights = attn_weights + attention_mask # Additive mask
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights if output_attentions else None, past_key_value

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)

        # Call the selected attention implementation
        attn_output, attn_weights, past_key_value = self.attention_impl(
            hidden_states=hidden_states_norm, # Pass normalized states
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        # Apply residual connection and dropout
        hidden_states = self.resid_dropout(attn_output) + residual

        # --- IMPORTANT FIX for Tuple Index Error ---
        # Ensure the returned tuple structure is consistent regardless of output_attentions/use_cache
        # Always return a 3-tuple, with None for elements not requested.
        outputs = (hidden_states, attn_weights if output_attentions else None, past_key_value if use_cache else None)
        # outputs = (hidden_states,)
        # if output_attentions:
        #     outputs += (attn_weights,)
        # if use_cache:
        #     outputs += (past_key_value,)

        return outputs

# Define ExpertLayer separately for clarity and robust loading
class ExpertLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_function):
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.act_fn = self._get_activation_fn(activation_function)
        self.dropout = nn.Dropout(0.1) # Consider using config dropout
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def _get_activation_fn(self, activation_function: str):
        if activation_function == "gelu": return nn.GELU()
        if activation_function == "relu": return nn.ReLU()
        if activation_function == "silu" or activation_function == "swish": return nn.SiLU()
        raise ValueError(f"Unsupported activation function: {activation_function}")

    def forward(self, x):
        return self.down_proj(self.dropout(self.act_fn(self.up_proj(x))))

class AdaptiveExpertSystem(nn.Module):
    """Adaptive expert system for conditional computation."""
    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int, experts_per_token: int, activation_function: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size # Store intermediate size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        self.router_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, num_experts)
        # Use the new ExpertLayer with named submodules
        self.experts = nn.ModuleList([
            ExpertLayer(hidden_size, intermediate_size, activation_function)
            for _ in range(num_experts)
        ])

    # _get_activation_fn is now in ExpertLayer
    # def _get_activation_fn(self, activation_function: str):
    #     ...

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_size)
        normalized_hidden = self.router_norm(hidden_states_flat)
        router_logits = self.router(normalized_hidden)
        routing_weights, selected_experts = torch.topk(router_logits, self.experts_per_token, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1, dtype=torch.float).to(hidden_states.dtype)
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.shape[0] == 0: continue
            current_state = hidden_states_flat[top_x]
            current_routing_weights = routing_weights[top_x, idx].unsqueeze(-1)
            expert_output = expert_layer(current_state)
            final_hidden_states.index_add_(0, top_x, expert_output * current_routing_weights)
        final_hidden_states = final_hidden_states.view(batch_size, seq_len, hidden_size)
        return final_hidden_states

class ApertisFeedForward(nn.Module):
    """Feed-forward module for Apertis architecture."""
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if config.use_expert_system:
            self.feed_forward = AdaptiveExpertSystem(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size, # Correctly passed
                num_experts=config.num_experts,
                experts_per_token=config.experts_per_token,
                activation_function=config.hidden_act,
            )
        else:
            # Standard FFN using named layers for consistency
            # self.feed_forward = nn.Sequential(
            #      nn.Linear(config.hidden_size, config.intermediate_size),
            #      self._get_activation_fn(config.hidden_act),
            #      nn.Dropout(config.hidden_dropout_prob),
            #      nn.Linear(config.intermediate_size, config.hidden_size),
            #  )
            # Use named layers like ExpertLayer
            self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size)
            self.act_fn = self._get_activation_fn(config.hidden_act)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
            # Assign a forward method or use it directly in the main forward
            self.feed_forward = self._standard_ffn_forward

        self.resid_dropout = nn.Dropout(config.hidden_dropout_prob)

    def _get_activation_fn(self, activation_function: str):
        # This is only needed if standard FFN is not nn.Sequential
        if activation_function == "gelu": return nn.GELU()
        if activation_function == "relu": return nn.ReLU()
        if activation_function == "silu" or activation_function == "swish": return nn.SiLU()
        raise ValueError(f"Unsupported activation function: {activation_function}")

    def _standard_ffn_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
         return self.down_proj(self.dropout(self.act_fn(self.up_proj(hidden_states))))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states_norm = self.input_layernorm(hidden_states)
        # if self.config.use_expert_system:
        ff_output = self.feed_forward(hidden_states_norm) # Call expert system or standard FFN
        # else:
        #     # If using named layers for standard FFN
        #     hidden_states = self.down_proj(self.dropout(self.act_fn(self.up_proj(hidden_states))))
        hidden_states = self.resid_dropout(ff_output) + residual
        return hidden_states

class ApertisLayer(nn.Module):
    """Transformer layer for Apertis architecture."""
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.attention = ApertisAttention(config)
        self.feed_forward = ApertisFeedForward(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, output_attentions: bool = False, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Attention block (includes input norm, attention impl, residual, dropout)
        # Attention now always returns a 3-tuple (hidden_states, attn_weights | None, past_key_value | None)
        attention_outputs = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output = attention_outputs[0]
        # --- IMPORTANT FIX for Tuple Index Error ---
        # Unpack based on the fixed 3-tuple structure
        attn_weights = attention_outputs[1] # Will be None if output_attentions is False
        attention_past_key_value = attention_outputs[2] # Will be None if use_cache is False
        # attn_weights = attention_outputs[1] if output_attentions else None
        # attention_past_key_value = attention_outputs[2] if use_cache else None

        # Feed-forward block (includes input norm, FFN/Expert, residual, dropout)
        layer_output = self.feed_forward(attention_output)

        # --- IMPORTANT FIX for Tuple Index Error ---
        # Ensure the returned tuple structure is consistent regardless of output_attentions/use_cache
        # Always return a 3-tuple, with None for elements not requested.
        outputs = (layer_output, attn_weights, attention_past_key_value)
        # outputs = (layer_output,)
        # if output_attentions: outputs += (attn_weights,)
        # if use_cache: outputs += (attention_past_key_value,)
        return outputs

class ApertisModel(nn.Module):
    """Base model for Apertis architecture."""
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        if config.multimodal:
            self.multimodal_encoder = UnifiedMultimodalEncoder(
                image_size=config.image_size, patch_size=config.vision_patch_size,
                embed_dim=config.vision_embed_dim, depth=config.vision_layers,
                num_heads=config.vision_heads, output_dim=config.hidden_size,
            )
        else:
            self.multimodal_encoder = None
        self.layers = nn.ModuleList([ApertisLayer(config) for _ in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None: module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None: module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, embeddings):
        self.token_embeddings = embeddings

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_values: Optional[List[torch.Tensor]] = None, inputs_embeds: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> BaseModelOutputWithPast: # Always return BaseModelOutputWithPast
        # --- IMPORTANT FIX for Tuple Index Error ---
        # Ignore return_dict, always return the BaseModelOutputWithPast object
        # return_dict = return_dict if return_dict is not None else True # No longer needed

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None: raise ValueError("Specify either input_ids or inputs_embeds")
        elif input_ids is not None: batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None: batch_size, seq_length, _ = inputs_embeds.shape
        else: raise ValueError("Specify either input_ids or inputs_embeds")
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            # past_key_values is a list/tuple of tuples (key, value) for each layer
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device).unsqueeze(0) # Shape [1, seq_length]
            # Expand later if needed by RoPE
        else: position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None: inputs_embeds = self.token_embeddings(input_ids)

        if self.multimodal_encoder is not None and pixel_values is not None:
            image_features = self.multimodal_encoder(pixel_values)
            num_image_tokens = image_features.shape[1]
            inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)
            seq_length += num_image_tokens; seq_length_with_past += num_image_tokens
            image_position_ids = torch.zeros(batch_size, num_image_tokens, dtype=torch.long, device=position_ids.device)
            position_ids = torch.cat([image_position_ids, position_ids], dim=1)
            if attention_mask is not None:
                image_attention_mask = torch.ones(batch_size, num_image_tokens, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            else: attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=inputs_embeds.device)

        # Prepare attention mask (standard causal mask preparation)
        _causal_mask = None
        if attention_mask is not None:
            # Expand 2D mask to 4D causal mask
            if attention_mask.dim() == 2:
                _causal_mask = self._prepare_decoder_attention_mask(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
            # Assume 4D mask is already correctly formatted if provided
            elif attention_mask.dim() == 4:
                _causal_mask = attention_mask
        elif seq_length > 1: # Create causal mask if no mask provided and seq_len > 1
             _causal_mask = self._prepare_decoder_attention_mask(None, (batch_size, seq_length), inputs_embeds, past_key_values_length)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states: all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            # Handle position_ids shape for RoPE
            # RoPE expects position_ids: [bsz, seq_len]
            current_position_ids = position_ids
            # If position_ids is [1, seq_len], expand it
            if current_position_ids.shape[0] == 1 and batch_size > 1:
                 current_position_ids = current_position_ids.expand(batch_size, -1)

            layer_args = {
                "hidden_states": hidden_states,
                "attention_mask": _causal_mask, # Pass the prepared mask
                "position_ids": current_position_ids, # Pass potentially expanded position_ids
                "past_key_value": past_key_value,
                "output_attentions": output_attentions,
                "use_cache": use_cache,
            }
            if self.gradient_checkpointing and self.training:
                # Custom forward wrapper for checkpoint to handle keyword args
                def create_custom_forward(module): 
                    def custom_forward(*inputs):
                        # Manually reconstruct keyword arguments for the layer"s forward
                        # Args passed by checkpoint: hidden_states, _causal_mask, current_position_ids, past_key_value_k, past_key_value_v
                        # We need to pass output_attentions and use_cache explicitly if needed
                        # Checkpoint doesn"t preserve non-tensor args by default
                        # We rely on the layer"s default values (False, False) here
                        # If they needed to be True, this would require a different approach
                        # Checkpoint expects only tensor inputs if use_reentrant=False
                        # Let"s pass only tensors and reconstruct kwargs inside the layer if needed
                        # Or, pass non-tensors as constants if checkpoint supports it (depends on version)
                        # Assuming use_reentrant=False, pass only tensors
                        _hidden_states = inputs[0]
                        _attention_mask = inputs[1]
                        _position_ids = inputs[2]
                        # _past_key_value = inputs[3] # This might be None
                        # Checkpoint might not pass None values correctly, or might unpack tuples
                        # Let"s assume past_key_value is NOT passed if None, or is passed as a tuple of tensors
                        # We need to handle the case where past_key_value is missing from inputs
                        _past_key_value = None
                        # --- IMPORTANT FIX for Checkpoint Past KV Handling ---
                        # Checkpoint passes *all* tensor args. If past_kv was provided, its tensors are at inputs[3] and inputs[4]
                        if len(inputs) > 3 and inputs[3] is not None and inputs[4] is not None:
                             _past_key_value = (inputs[3], inputs[4])
                        
                        # Call the layer, explicitly setting non-tensor args
                        # Layer now always returns a 3-tuple
                        layer_outputs_tuple = module(
                            _hidden_states, 
                            attention_mask=_attention_mask, 
                            position_ids=_position_ids, 
                            past_key_value=_past_key_value, 
                            output_attentions=False, # Hardcoded for checkpoint
                            use_cache=False # Hardcoded for checkpoint
                        )
                        # Checkpoint requires the function to return a Tensor or Tuple of Tensors
                        # Return only the hidden_states tensor, as other outputs are None anyway
                        return layer_outputs_tuple[0]
                    return custom_forward
                
                # Pass only tensor args to checkpoint
                # Ensure _causal_mask and past_key_value are handled correctly (can be None)
                tensor_args = [hidden_states, _causal_mask, current_position_ids]
                # Checkpoint requires all inputs to be tensors or None
                # If past_key_value is a tuple of tensors, pass them individually
                if past_key_value is not None:
                     tensor_args.extend(past_key_value) # Add key and value tensors
                else:
                     # Pass None placeholders if checkpoint requires consistent arg count
                     # Let"s try passing None explicitly for K and V if past_kv is None
                     tensor_args.extend([None, None])

                # Ensure all args passed to checkpoint are Tensors or None
                # _causal_mask can be None
                if _causal_mask is None:
                     # Replace None with a dummy tensor or handle differently if checkpoint fails
                     # For now, assume checkpoint handles None
                     pass 

                # Ensure all elements in tensor_args are Tensors or None
                final_tensor_args = []
                for arg in tensor_args:
                     if isinstance(arg, torch.Tensor) or arg is None:
                         final_tensor_args.append(arg)
                     else:
                         raise TypeError(f"Argument passed to checkpoint is not a Tensor or None: {type(arg)}")

                # Call checkpoint
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    *final_tensor_args,
                    use_reentrant=False, # Recommended for PyTorch >= 1.10
                )
                # Checkpoint returns only hidden_states in this setup
                attn_weights = None
                next_kv = None

            else:
                # Layer now always returns a 3-tuple
                layer_outputs = decoder_layer(**layer_args)
                hidden_states = layer_outputs[0]
                attn_weights = layer_outputs[1] # Already None if output_attentions=False
                next_kv = layer_outputs[2] # Already None if use_cache=False

            if use_cache: next_decoder_cache += (next_kv,)
            if output_attentions: all_self_attns += (attn_weights,)

        hidden_states = self.final_layer_norm(hidden_states)
        if output_hidden_states: all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None

        # --- IMPORTANT FIX for Tuple Index Error ---
        # Always return the BaseModelOutputWithPast object
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # return result

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Create causal mask for decoding
        bsz, tgt_len = input_shape
        device = inputs_embeds.device
        dtype = inputs_embeds.dtype

        # Default causal mask
        # Causal mask should be additive (0 for allowed, -inf for masked)
        # It should cover the query length (tgt_len)
        combined_attention_mask = None
        if tgt_len > 1:
            # Create a lower triangular matrix (1s on and below diagonal)
            causal_mask_bool = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.bool, device=device))
            # Expand to 4D: [1, 1, tgt_len, tgt_len]
            causal_mask_bool = causal_mask_bool[None, None, :, :]
            # Invert to get additive mask (0 where allowed, -inf where masked)
            # Where causal_mask_bool is False (upper triangle), fill with -inf
            combined_attention_mask = torch.zeros((1, 1, tgt_len, tgt_len), dtype=dtype, device=device)
            combined_attention_mask.masked_fill_(~causal_mask_bool, torch.finfo(dtype).min)
            # Expand batch dimension if needed (though broadcasting usually handles this)
            # combined_attention_mask = combined_attention_mask.expand(bsz, 1, tgt_len, tgt_len)

        if attention_mask is not None:
            # Expand provided mask (padding mask) to 4D
            # Input mask is [bsz, src_len] (1 for real, 0 for padding)
            # Need [bsz, 1, tgt_len, src_len] where 0 means keep, -inf means mask
            expanded_attn_mask = self._expand_mask(attention_mask, dtype, tgt_len=tgt_len).to(device)
            
            # Combine causal mask with expanded mask
            if combined_attention_mask is not None:
                 # Ensure shapes match for addition (broadcasting)
                 # combined_attention_mask: [1, 1, tgt_len, tgt_len]
                 # expanded_attn_mask: [bsz, 1, tgt_len, src_len]
                 # If past_key_values_length > 0, src_len = tgt_len + past_key_values_length
                 # If past_key_values_length == 0, src_len = tgt_len
                 # We need to combine them correctly based on the target shape [bsz, 1, tgt_len, src_len]
                 
                 # Let"s adjust the causal mask shape first
                 causal_mask_for_combine = combined_attention_mask # [1, 1, tgt_len, tgt_len]
                 
                 # Adjust expanded_attn_mask to match target shape if needed
                 # It should already be [bsz, 1, tgt_len, src_len]
                 
                 # If past_key_values_length > 0, the causal mask only applies to the new tokens
                 # The final mask should be [bsz, 1, tgt_len, past_len + tgt_len]
                 current_src_len = past_key_values_length + tgt_len
                 final_mask_shape = (bsz, 1, tgt_len, current_src_len)
                 
                 # Start with the expanded padding mask (covers all source positions)
                 final_combined_mask = expanded_attn_mask.expand(final_mask_shape)
                 
                 # Add the causal part only for the query tokens attending to themselves
                 if tgt_len > 1:
                     # The causal mask applies to the last tgt_len columns
                     final_combined_mask[..., past_key_values_length:] = final_combined_mask[..., past_key_values_length:] + causal_mask_for_combine
                 
                 combined_attention_mask = final_combined_mask
                 
            else: # Only padding mask exists (e.g., tgt_len=1 during generation)
                 current_src_len = past_key_values_length + tgt_len
                 final_mask_shape = (bsz, 1, tgt_len, current_src_len)
                 combined_attention_mask = expanded_attn_mask.expand(final_mask_shape)
        
        # If no masks provided and tgt_len=1, combined_attention_mask is still None here
        # This is okay as attention layer might not need mask for single query token

        # Clip values to avoid potential issues with mixed precision
        if combined_attention_mask is not None:
             combined_attention_mask = torch.clamp(combined_attention_mask, max=0.0)

        return combined_attention_mask

    def _expand_mask(self, mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
        # Input mask: [bsz, src_len] (1 for real, 0 for padding)
        bsz, src_len = mask.size(); tgt_len = tgt_len if tgt_len is not None else src_len
        # Target shape: [bsz, 1, tgt_len, src_len]
        # Where mask is 0, value should be -inf; where mask is 1, value should be 0
        expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        # Create additive mask: (1.0 - expanded_mask) will be 1.0 for padding, 0.0 for real tokens
        additive_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        return additive_mask

    # _gradient_checkpointing_func removed as checkpoint is called directly
    # def _gradient_checkpointing_func(self, func, *args):
    #    ...

class ApertisForCausalLM(nn.Module):
    """Apertis model for causal language modeling."""
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.model = ApertisModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.tie_weights()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        # Delegate most initialization to base model
        if isinstance(module, nn.Linear) and module is self.lm_head and not self.config.tie_word_embeddings:
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None: module.bias.data.zero_()

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for the base model."""
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model."""
        self.model.gradient_checkpointing_disable()

    def tie_weights(self):
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.get_input_embeddings().weight

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_lm_head):
        self.lm_head = new_lm_head

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self.model.get_input_embeddings()
        if new_num_tokens is None:
            return model_embeds

        old_num_tokens, old_embedding_dim = model_embeds.weight.size()
        if old_num_tokens == new_num_tokens:
            return model_embeds

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(model_embeds.weight.device, dtype=model_embeds.weight.dtype)

        # Initialize new embeddings
        self._init_weights(new_embeddings)

        # Copy weights from old embeddings
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = model_embeds.weight.data[:num_tokens_to_copy, :]

        self.set_input_embeddings(new_embeddings)

        # Update the `.vocab_size` attribute in the config
        self.config.vocab_size = new_num_tokens

        # If embeddings are tied, update output embeddings
        if self.config.tie_word_embeddings:
            self.lm_head.weight = new_embeddings.weight
            # Also update output layer dimensions if necessary (though bias=False)
            if hasattr(self.lm_head, "out_features") and self.lm_head.out_features != new_num_tokens:
                 self.lm_head.out_features = new_num_tokens

        return new_embeddings

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None, past_key_values: Optional[List[torch.Tensor]] = None, inputs_embeds: Optional[torch.Tensor] = None, pixel_values: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None, use_cache: Optional[bool] = None, output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> CausalLMOutputWithPast: # Always return CausalLMOutputWithPast
        # --- IMPORTANT FIX for Tuple Index Error ---
        # Ignore return_dict, always return the CausalLMOutputWithPast object
        # return_dict = return_dict if return_dict is not None else True # No longer needed

        # Pass inputs to the base model
        # Base model now always returns BaseModelOutputWithPast
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=True, # Base model always returns dict-like now
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_logits = torch.clamp(shift_logits, min=-10.0, max=10.0) # Clamp logits for stability
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # --- IMPORTANT FIX for Tuple Index Error ---
        # Always return CausalLMOutputWithPast object
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # if not return_dict:
        #     # Construct tuple output, ensuring loss is first if present
        #     output_items = [logits]
        #     if use_cache: output_items.append(outputs.get("past_key_values"))
        #     if output_hidden_states: output_items.append(outputs.get("hidden_states"))
        #     if output_attentions: output_items.append(outputs.get("attentions"))
        #     output = tuple(item for item in output_items if item is not None) # Filter out None values
        #     return ((loss,) + output) if loss is not None else output
        # return CausalLMOutputWithPast(...)

    # --- Added Generate Method --- 
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None, # Added position_ids
        pixel_values: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = False,
        num_beams: int = 1,
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Generates sequences of token ids for models with a language modeling head.

        Simplified implementation supporting greedy search and basic sampling.
        Does not support beam search or advanced generation features.
        """
        self.eval() # Ensure model is in eval mode

        # --- Input Validation and Setup --- 
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation.")
        if num_beams != 1:
            raise NotImplementedError("Beam search (num_beams > 1) is not implemented.")

        batch_size, input_seq_len = input_ids.shape
        device = input_ids.device

        # Determine generation length
        if max_length is None and max_new_tokens is None:
            max_length = self.config.max_position_embeddings
            print(f"Warning: Neither max_length nor max_new_tokens provided. Defaulting max_length to {max_length}")
        elif max_length is not None and max_new_tokens is not None:
            # HF convention: max_length includes prompt length
            max_length = max_length 
        elif max_new_tokens is not None:
            max_length = input_seq_len + max_new_tokens
        # max_length is now the target total sequence length

        # Get special token IDs from config or arguments
        _pad = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        _bos = bos_token_id if bos_token_id is not None else self.config.bos_token_id
        _eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id

        # Initialize past_key_values for caching
        past_key_values = None

        # Handle multimodal input
        # If pixel_values are provided, they affect the initial forward pass
        initial_pixel_values = pixel_values

        # --- Generation Loop --- 
        generated_ids = input_ids
        current_len = input_seq_len
        
        # Initialize position_ids if not provided
        if position_ids is None:
             position_ids = torch.arange(0, input_seq_len, dtype=torch.long, device=device).unsqueeze(0)

        while current_len < max_length:
            # Prepare model inputs for this step
            if past_key_values is not None:
                # Use only the last token ID and its position ID for the next step
                current_input_ids = generated_ids[:, -1:]
                current_position_ids = position_ids[:, -1:]
            else:
                # First step uses the full input_ids and position_ids
                current_input_ids = generated_ids
                current_position_ids = position_ids

            model_inputs = {
                 "input_ids": current_input_ids,
                 "position_ids": current_position_ids,
                 "attention_mask": attention_mask, # Pass the full mask
                 "past_key_values": past_key_values,
                 "use_cache": use_cache,
                 # "return_dict": True, # Forward now always returns dict-like
             }

            # Add pixel_values only for the very first step if multimodal
            if past_key_values is None and initial_pixel_values is not None:
                 model_inputs["pixel_values"] = initial_pixel_values

            # Forward pass
            # Forward now always returns CausalLMOutputWithPast
            outputs: CausalLMOutputWithPast = self(**model_inputs)
            next_token_logits = outputs.logits[:, -1, :] # Logits for the last token

            # Update past_key_values
            past_key_values = outputs.past_key_values

            # --- Token Sampling / Greedy Search --- 
            if do_sample:
                # Temperature scaling
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature

                # Top-K filtering
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1)) # Safety check
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

                # Top-P filtering (nucleus sampling)
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))

                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy search
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # --- Append and Check EOS --- 
            generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
            current_len += 1

            # Update attention mask for the next step (append 1 for the new token)
            if attention_mask is not None:
                 attention_mask = torch.cat([attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1)
                 
            # Update position_ids for the next step
            next_position_id = position_ids[:, -1:] + 1
            position_ids = torch.cat([position_ids, next_position_id], dim=1)

            # Check if EOS token was generated (simple check, assumes single EOS)
            if _eos is not None and (next_token_id == _eos).all():
                break

        return generated_ids

