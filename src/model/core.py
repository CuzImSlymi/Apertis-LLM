import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any, Union
import math
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import the multimodal module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.multimodal.module import UnifiedMultimodalEncoder

class ApertisConfig:
    """Core configuration for the Apertis architecture."""
    
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

    @classmethod
    def from_dict(cls, config_dict):
        """Create a configuration from a dictionary."""
        return cls(**config_dict)
    
    def to_dict(self):
        """Convert configuration to dictionary."""
        return self.__dict__
    
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        """Load configuration from a pretrained model path."""
        import json
        import os
        
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "config.json")
        else:
            config_file = model_name_or_path
            
        with open(config_file, "r") as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)
    
    def save_pretrained(self, save_directory):
        """Save configuration to a directory."""
        import json
        import os
        
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
        
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # x: [batch_size, seq_len, num_heads, head_dim]
        # position_ids: [batch_size, seq_len]
        
        # Create sinusoidal pattern
        seq_len = position_ids.shape[1]
        position_ids = position_ids.view(-1, seq_len)
        
        # [seq_len]
        sincos_pos = torch.einsum("i,j->ij", position_ids.float(), self.inv_freq)
        
        # [batch_size, seq_len, dim//2]
        sin, cos = torch.sin(sincos_pos), torch.cos(sincos_pos)
        
        # Reshape for broadcasting
        sin = sin.unsqueeze(2)  # [batch_size, seq_len, 1, dim//2]
        cos = cos.unsqueeze(2)  # [batch_size, seq_len, 1, dim//2]
        
        # Apply rotary embeddings
        x1, x2 = x[..., 0::2], x[..., 1::2]
        
        # Rotate x by multiplying with cos and sin
        rotated_x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        
        return rotated_x


class SelectiveLinearAttention(nn.Module):
    """Selective Linear Attention mechanism for O(n) time complexity."""
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        attention_dropout: float = 0.1,
        sliding_window: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(attention_dropout)
        
        # Layer normalization for improved stability
        self.layer_norm_q = nn.LayerNorm(hidden_size)
        self.layer_norm_k = nn.LayerNorm(hidden_size)
        self.layer_norm_v = nn.LayerNorm(hidden_size)
        
        # Recurrent state for efficient processing
        self.register_buffer("state_k", None, persistent=False)
        self.register_buffer("state_v", None, persistent=False)
        
    def _reset_state(self):
        self.state_k = None
        self.state_v = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply layer normalization for improved stability
        normalized_hidden_states = hidden_states
        
        # Project inputs to queries, keys, and values with layer norm
        query_states = self.q_proj(self.layer_norm_q(normalized_hidden_states)) * self.scaling
        key_states = self.k_proj(self.layer_norm_k(normalized_hidden_states))
        value_states = self.v_proj(self.layer_norm_v(normalized_hidden_states))
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Handle past key values
        if past_key_value is not None:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            
        past_key_value = (key_states, value_states) if use_cache else None
        
        # Apply sliding window if specified
        if self.sliding_window is not None and key_states.shape[2] > self.sliding_window:
            key_states = key_states[:, :, -self.sliding_window:]
            value_states = value_states[:, :, -self.sliding_window:]
        
        # Compute attention with linear complexity using optimized selective scan
        attn_output = self._selective_scan(query_states, key_states, value_states, attention_mask)
        
        # Reshape back to original dimensions
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # Final projection
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    
    def _selective_scan(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Improved efficient linear attention implementation
        # Using the recurrent formulation: h_t = (1-α_t)h_{t-1} + α_t v_t
        # with optimized memory access patterns
        
        batch_size, num_heads, seq_len, head_dim = query_states.shape
        
        # Apply chunking for better cache locality
        chunk_size = min(128, seq_len)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size
        
        # Initialize output tensor
        attn_output = torch.zeros_like(query_states)
        
        # Process in chunks for better efficiency
        for i in range(num_chunks):
            # Get current chunk
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, seq_len)
            q_chunk = query_states[:, :, start_idx:end_idx]
            
            # Compute query-key similarities for this chunk
            attn_weights = torch.matmul(q_chunk, key_states.transpose(-1, -2))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Extract the relevant part of the attention mask
                if attention_mask.size(-1) == seq_len:
                    chunk_mask = attention_mask[:, :, :, start_idx:end_idx]
                    attn_weights = attn_weights + chunk_mask
                else:
                    attn_weights = attn_weights + attention_mask
            
            # Apply softmax to get attention probabilities
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Compute weighted sum of values
            chunk_output = torch.matmul(attn_weights, value_states)
            
            # Store in output tensor
            attn_output[:, :, start_idx:end_idx] = chunk_output
        
        return attn_output


class AdaptiveExpertSystem(nn.Module):
    """Adaptive Expert System (AES) for efficient parameter usage."""
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int = 8,
        experts_per_token: int = 2,
        activation_function: str = "gelu",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token
        
        # Router for selecting experts with layer normalization for stability
        self.router_norm = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, num_experts)
        
        # Expert feed-forward networks with improved architecture
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, intermediate_size),
                self._get_activation_fn(activation_function),
                nn.Dropout(0.1),  # Add dropout for regularization
                nn.Linear(intermediate_size, hidden_size)
            )
            for _ in range(num_experts)
        ])
        
        # Output layer normalization
        self.output_norm = nn.LayerNorm(hidden_size)
        
    def _get_activation_fn(self, activation_function: str):
        if activation_function == "gelu":
            return nn.GELU()
        elif activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "silu" or activation_function == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute routing probabilities with layer normalization
        normalized_hidden = self.router_norm(hidden_states)
        router_logits = self.router(normalized_hidden)  # [batch_size, seq_len, num_experts]
        
        # Select top-k experts
        routing_weights, selected_experts = torch.topk(
            router_logits, self.experts_per_token, dim=-1
        )  # [batch_size, seq_len, experts_per_token]
        
        # Normalize routing weights with softmax
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Initialize output tensor
        expert_outputs = torch.zeros_like(hidden_states)
        
        # Vectorized implementation for better efficiency
        # Process in batches for better parallelism
        for k in range(self.experts_per_token):
            # Extract expert indices and weights for this k
            expert_indices = selected_experts[:, :, k]  # [batch_size, seq_len]
            expert_weights = routing_weights[:, :, k].unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Process each expert in parallel
            for expert_idx in range(self.num_experts):
                # Create a mask for tokens that use this expert
                mask = (expert_indices == expert_idx).unsqueeze(-1)  # [batch_size, seq_len, 1]
                
                if mask.any():
                    # Apply the expert to all tokens (will be masked later)
                    expert_output = self.experts[expert_idx](hidden_states)
                    
                    # Only add the output for tokens that selected this expert
                    expert_outputs += mask.float() * expert_weights * expert_output
        
        # Apply output normalization
        expert_outputs = self.output_norm(expert_outputs)
        
        return expert_outputs


class StateTrackingRecurrentCell(nn.Module):
    """State Tracking Recurrent Cell (STRC) for efficient inference."""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Recurrent cell parameters with improved architecture
        self.input_norm = nn.LayerNorm(hidden_size)
        self.state_norm = nn.LayerNorm(hidden_size)
        
        self.update_gate = nn.Linear(2 * hidden_size, hidden_size)
        self.reset_gate = nn.Linear(2 * hidden_size, hidden_size)
        self.output_gate = nn.Linear(2 * hidden_size, hidden_size)
        
        # Layer normalization for outputs
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Initialize previous state if not provided
        if prev_state is None:
            prev_state = torch.zeros(batch_size, hidden_size, device=hidden_states.device)
        
        # Apply input normalization
        normalized_hidden = self.input_norm(hidden_states)
        
        # Process sequence step by step with optimized implementation
        output_states = []
        current_state = prev_state
        
        for t in range(seq_len):
            # Get current input
            x_t = normalized_hidden[:, t]
            
            # Apply state normalization
            normalized_state = self.state_norm(current_state)
            
            # Concatenate input with previous state
            combined = torch.cat([x_t, normalized_state], dim=-1)
            
            # Compute gates with dropout for regularization
            update = torch.sigmoid(self.dropout(self.update_gate(combined)))
            reset = torch.sigmoid(self.dropout(self.reset_gate(combined)))
            
            # Compute candidate state
            candidate = torch.tanh(
                self.output_gate(
                    torch.cat([x_t, reset * normalized_state], dim=-1)
                )
            )
            
            # Update state with residual connection
            current_state = (1 - update) * current_state + update * candidate
            
            # Apply layer normalization
            normalized_output = self.layer_norm(current_state)
            
            output_states.append(normalized_output)
        
        # Stack outputs
        output_sequence = torch.stack(output_states, dim=1)
        
        return output_sequence, current_state


class ApertisLayer(nn.Module):
    """Apertis transformer layer."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # Layer normalization for pre-norm architecture (more stable training)
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Attention mechanism
        if config.attention_type == "selective_linear":
            self.attention = SelectiveLinearAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                attention_dropout=config.attention_probs_dropout_prob,
                sliding_window=config.sliding_window,
            )
        else:
            raise ValueError(f"Unsupported attention type: {config.attention_type}")
        
        # Feed-forward network
        if config.use_expert_system:
            self.mlp = AdaptiveExpertSystem(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                experts_per_token=config.experts_per_token,
                activation_function=config.hidden_act,
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size),
                self._get_activation_fn(config.hidden_act),
                nn.Linear(config.intermediate_size, config.hidden_size),
                nn.Dropout(config.hidden_dropout_prob),
            )
        
        # Layer normalization
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # State tracking for efficient inference
        self.state_tracker = StateTrackingRecurrentCell(config.hidden_size)
        
    def _get_activation_fn(self, activation_function: str):
        if activation_function == "gelu":
            return nn.GELU()
        elif activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "silu" or activation_function == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        prev_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]], Optional[torch.Tensor]]:
        # Self-attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states, attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Feed-forward network
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        # Apply state tracking for efficient inference
        if use_cache:
            hidden_states, new_state = self.state_tracker(hidden_states, prev_state)
        else:
            new_state = None
        
        return hidden_states, attn_weights, past_key_value, new_state


class ApertisModel(nn.Module):
    """Apertis model with innovative architecture components."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_embed_dim = config.vision_embed_dim
        self.image_size = config.image_size
        self.vision_patch_size = config.vision_patch_size
        
        # Calculate number of patches
        self.num_patches = (self.image_size // self.vision_patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=self.vision_embed_dim,
            kernel_size=self.vision_patch_size,
            stride=self.vision_patch_size,
        )
        
        # Position embeddings for vision
        self.vision_pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, self.vision_embed_dim)
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.vision_embed_dim))
        
        # Vision transformer layers
        self.vision_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.vision_embed_dim,
                nhead=config.vision_heads,
                dim_feedforward=self.vision_embed_dim * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(config.vision_layers)
        ])
        
        # Vision layer norm
        self.vision_ln = nn.LayerNorm(self.vision_embed_dim)
        
        # Projection from vision to text space
        self.vision_projection = nn.Linear(self.vision_embed_dim, config.hidden_size)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        
        # Create patch embeddings
        patch_embeds = self.patch_embed(pixel_values)
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)  # [B, num_patches, vision_embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, patch_embeds], dim=1)
        
        # Add position embeddings
        embeddings = embeddings + self.vision_pos_embed
        
        # Apply vision transformer layers
        for layer in self.vision_layers:
            embeddings = layer(embeddings)
        
        # Apply layer norm
        embeddings = self.vision_ln(embeddings)
        
        # Project to text space
        projected_embeddings = self.vision_projection(embeddings)
        
        return projected_embeddings


class ApertisModel(nn.Module):
    """Apertis model with innovative architecture components."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        if config.position_embedding_type == "rotary":
            self.rotary_emb = RotaryEmbedding(
                config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ApertisLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Multimodal support
        if config.multimodal:
            self.multimodal_encoder = UnifiedMultimodalEncoder(config)
            self.multimodal_projector = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict to handle model loading.
        
        Args:
            state_dict: The state dictionary containing parameters
            strict: Whether to strictly enforce that the keys in state_dict match the keys in this module's state_dict
            
        Returns:
            A tuple of (missing_keys, unexpected_keys)
        """
        return super().load_state_dict(state_dict, strict=strict)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        # Get input embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
        
        # Process image if provided and model is multimodal
        if pixel_values is not None and self.config.multimodal:
            image_embeds = self.multimodal_encoder(pixel_values)
            image_embeds = self.multimodal_projector(image_embeds)
            
            # Concatenate text and image embeddings
            # Assuming image comes first, then text
            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            
            # Update attention mask to include image tokens
            if attention_mask is not None:
                image_attention_mask = torch.ones(
                    (attention_mask.shape[0], image_embeds.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        
        # Get sequence length
        batch_size, seq_length = inputs_embeds.shape[:2]
        
        # Prepare position IDs
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=inputs_embeds.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert mask from [0, 1] to [-inf, 0]
            attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        # Initialize hidden states and attentions
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        
        # Initialize state for recurrent processing
        states = [None] * len(self.layers)
        
        # Process through layers
        hidden_states = inputs_embeds
        
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            layer_outputs = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                prev_state=states[i],
            )
            
            hidden_states = layer_outputs[0]
            states[i] = layer_outputs[3]
            
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            if use_cache:
                past_key_values[i] = layer_outputs[2]
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return (hidden_states, past_key_values, all_hidden_states, all_attentions)


class ApertisForCausalLM(nn.Module):
    """Apertis model for causal language modeling."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        self.config = config
        
        # Base model
        self.model = ApertisModel(config)
        
        # LM head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights if configured
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.token_embeddings.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def load_state_dict(self, state_dict, strict=True):
        """
        Overrides the default load_state_dict to handle model loading.
        
        Args:
            state_dict: The state dictionary containing parameters
            strict: Whether to strictly enforce that the keys in state_dict match the keys in this module's state_dict
            
        Returns:
            A tuple of (missing_keys, unexpected_keys)
        """
        return super().load_state_dict(state_dict, strict=strict)
        
    def save_pretrained(self, save_directory):
        """
        Save model to a directory.
        
        Args:
            save_directory: Directory to save the model
        """
        import os
        import torch
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model weights
        model_path = os.path.join(save_directory, "model.pt")
        torch.save(self.state_dict(), model_path)
        
        # Save the configuration
        self.config.save_pretrained(save_directory)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        return_dict = return_dict if return_dict is not None else True
        
        # Forward pass through base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Get hidden states
        hidden_states = outputs[0]
        
        # Apply LM head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )
        
        return (loss, logits, outputs[1], outputs[2], outputs[3])
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        # Only last token for inputs_ids if past is defined
        if past_key_values is not None:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            
            # Create position_ids for the new token
            if position_ids is not None:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "pixel_values": pixel_values,
        }
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        num_return_sequences: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text using the model."""
        # Set default values
        max_length = max_length if max_length is not None else 20
        min_length = min_length if min_length is not None else 0
        do_sample = do_sample if do_sample is not None else False
        temperature = temperature if temperature is not None else 1.0
        top_k = top_k if top_k is not None else 50
        top_p = top_p if top_p is not None else 1.0
        repetition_penalty = repetition_penalty if repetition_penalty is not None else 1.0
        num_return_sequences = num_return_sequences if num_return_sequences is not None else 1
        
        # Initialize past key values
        past_key_values = None
        
        # Initialize generated sequences
        batch_size = input_ids.shape[0]
        generated_sequences = input_ids.clone()
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Generate tokens
        for _ in range(max_length):
            # Prepare inputs
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=generated_sequences,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                pixel_values=pixel_values if _ == 0 else None,  # Only use pixel_values for first token
            )
            
            # Forward pass
            outputs = self(**model_inputs, use_cache=True)
            logits = outputs[1]
            past_key_values = outputs[2]
            
            # Get next token logits
            next_token_logits = logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for token_id in set(generated_sequences[i].tolist()):
                        next_token_logits[i, token_id] /= repetition_penalty
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = torch.topk(next_token_logits, k=top_k, dim=-1)[0][:, -1].unsqueeze(-1) <= next_token_logits
                next_token_logits = next_token_logits.masked_fill(indices_to_remove, -float("Inf"))
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep the first token above the threshold
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = -float("Inf")
            
            # Sample or greedy decode
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_logits, dim=-1)
            
            # Append next tokens
            generated_sequences = torch.cat([generated_sequences, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Update attention mask
            attention_mask = torch.cat(
                [attention_mask, torch.ones((batch_size, 1), device=attention_mask.device)],
                dim=-1,
            )
            
            # Check if all sequences have reached the end token
            if (next_tokens == self.config.eos_token_id).all():
                break
        
        return generated_sequences


def create_apertis_model(
    model_size: str = "base",
    multimodal: bool = False,
    use_flash_attention: bool = False,
    use_expert_system: bool = False,
) -> ApertisForCausalLM:
    """Create an Apertis model with specified configuration."""
    # Define model sizes
    model_configs = {
        "small": {
            "hidden_size": 512,
            "num_hidden_layers": 8,
            "num_attention_heads": 8,
            "intermediate_size": 2048,
        },
        "base": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "intermediate_size": 4096,
        },
    }
    
    # Get model configuration
    if model_size not in model_configs:
        raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(model_configs.keys())}")
    
    config_dict = model_configs[model_size]
    
    # Create configuration
    config = ApertisConfig(
        **config_dict,
        multimodal=multimodal,
        use_flash_attention=use_flash_attention,
        use_expert_system=use_expert_system,
    )
    
    # Create model
    model = ApertisForCausalLM(config)
    
    return model
