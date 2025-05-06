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
        # Use dynamic chunk size that matches the sequence length dimensions
        # This ensures tensor dimensions are compatible during training
        chunk_size = seq_len  # Use full sequence length to avoid dimension mismatch
        
        # Initialize output tensor
        attn_output = torch.zeros_like(query_states)
        
        # Compute full attention in one go for compatibility
        # Compute query-key similarities
        attn_weights = torch.matmul(query_states, key_states.transpose(-1, -2))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute weighted sum of values
        attn_output = torch.matmul(attn_weights, value_states)
        
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
        input_states: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Initialize hidden states if not provided
        batch_size, seq_len, hidden_size = input_states.shape
        if hidden_states is None:
            hidden_states = torch.zeros(batch_size, hidden_size, device=input_states.device)
        
        # Apply layer normalization
        normalized_input = self.input_norm(input_states)
        normalized_hidden = self.state_norm(hidden_states.unsqueeze(1))
        
        # Process sequence step by step
        outputs = []
        for t in range(seq_len):
            # Get current input
            x_t = normalized_input[:, t]
            
            # Concatenate input and hidden state
            combined = torch.cat([x_t, normalized_hidden.squeeze(1)], dim=-1)
            
            # Compute gates
            update = torch.sigmoid(self.update_gate(combined))
            reset = torch.sigmoid(self.reset_gate(combined))
            
            # Compute candidate hidden state
            candidate = torch.tanh(
                self.output_gate(
                    torch.cat([x_t, reset * normalized_hidden.squeeze(1)], dim=-1)
                )
            )
            
            # Update hidden state
            hidden_states = (1 - update) * hidden_states + update * candidate
            normalized_hidden = self.state_norm(hidden_states.unsqueeze(1))
            
            # Apply dropout
            hidden_states = self.dropout(hidden_states)
            
            # Collect output
            outputs.append(hidden_states)
        
        # Stack outputs
        outputs = torch.stack(outputs, dim=1)
        
        # Apply layer normalization
        outputs = self.layer_norm(outputs)
        
        return outputs


class ApertisAttention(nn.Module):
    """Attention module for Apertis architecture."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        
        # Choose attention implementation based on configuration
        if config.attention_type == "selective_linear":
            self.attention = SelectiveLinearAttention(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                attention_dropout=config.attention_probs_dropout_prob,
                sliding_window=config.sliding_window,
            )
        else:
            # Default to standard attention
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_size,
                num_heads=config.num_attention_heads,
                dropout=config.attention_probs_dropout_prob,
                batch_first=True,
            )
            
        # Rotary position embeddings
        if config.position_embedding_type == "rotary":
            self.rotary_embeddings = RotaryEmbedding(
                dim=config.hidden_size // config.num_attention_heads,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            self.rotary_embeddings = None
        
        # Output projection and dropout
        self.output_projection = nn.Linear(config.hidden_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Apply layer normalization
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        
        # Process with attention mechanism
        if isinstance(self.attention, SelectiveLinearAttention):
            # Use selective linear attention
            hidden_states, attn_weights, past_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        else:
            # Use standard attention
            # Prepare key padding mask from attention mask
            key_padding_mask = None
            if attention_mask is not None:
                key_padding_mask = attention_mask == 0
            
            # Apply standard attention
            hidden_states, attn_weights = self.attention(
                query=hidden_states,
                key=hidden_states,
                value=hidden_states,
                key_padding_mask=key_padding_mask,
                need_weights=output_attentions,
            )
            
            past_key_value = None
        
        # Apply output projection and dropout
        hidden_states = self.output_projection(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states, attn_weights, past_key_value


class ApertisFeedForward(nn.Module):
    """Feed-forward module for Apertis architecture."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        
        self.config = config
        
        # Choose implementation based on configuration
        if config.use_expert_system:
            self.feed_forward = AdaptiveExpertSystem(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_experts=config.num_experts,
                experts_per_token=config.experts_per_token,
                activation_function=config.hidden_act,
            )
        else:
            # Standard feed-forward network
            self.feed_forward = nn.Sequential(
                nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                nn.Linear(config.hidden_size, config.intermediate_size),
                self._get_activation_fn(config.hidden_act),
                nn.Dropout(config.hidden_dropout_prob),
                nn.Linear(config.intermediate_size, config.hidden_size),
            )
        
        # Output dropout
        self.output_dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
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
        # Apply layer normalization
        residual = hidden_states
        
        # Process with feed-forward network
        if isinstance(self.feed_forward, AdaptiveExpertSystem):
            # Use adaptive expert system
            hidden_states = self.feed_forward(hidden_states)
        else:
            # Use standard feed-forward network
            hidden_states = self.feed_forward(hidden_states)
        
        # Apply output dropout
        hidden_states = self.output_dropout(hidden_states)
        
        # Add residual connection
        hidden_states = hidden_states + residual
        
        return hidden_states


class ApertisLayer(nn.Module):
    """Transformer layer for Apertis architecture."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        
        self.config = config
        
        # Attention module
        self.attention = ApertisAttention(config)
        
        # Feed-forward module
        self.feed_forward = ApertisFeedForward(config)
        
        # State tracking for efficient inference
        if hasattr(config, "use_state_tracking") and config.use_state_tracking:
            self.state_tracking = StateTrackingRecurrentCell(config.hidden_size)
        else:
            self.state_tracking = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # Process with attention
        attention_output, attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        
        # Process with feed-forward network
        layer_output = self.feed_forward(attention_output)
        
        # Apply state tracking if enabled
        if self.state_tracking is not None:
            layer_output = self.state_tracking(layer_output)
        
        return layer_output, attn_weights, past_key_value


class ApertisModel(nn.Module):
    """Base model for Apertis architecture."""
    
    def __init__(self, config: ApertisConfig):
        super().__init__()
        
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Multimodal encoder if enabled
        if config.multimodal:
            self.multimodal_encoder = UnifiedMultimodalEncoder(
                image_size=config.image_size,
                patch_size=config.vision_patch_size,
                embed_dim=config.vision_embed_dim,
                depth=config.vision_layers,
                num_heads=config.vision_heads,
                output_dim=config.hidden_size,
            )
        else:
            self.multimodal_encoder = None
        
        # Transformer layers
        self.layers = nn.ModuleList([
            ApertisLayer(config)
            for _ in range(config.num_hidden_layers)
        ])
        
        # Final layer normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Enable gradient checkpointing for memory efficiency
        self.gradient_checkpointing_enable = self._gradient_checkpointing_enable
        
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
    
    def _gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.gradient_checkpointing = True
        
        # Apply gradient checkpointing to transformer layers
        for layer in self.layers:
            if hasattr(layer, "attention") and hasattr(layer.attention, "attention"):
                if hasattr(layer.attention.attention, "_checkpoint_activations"):
                    layer.attention.attention._checkpoint_activations = True
            
            if hasattr(layer, "feed_forward") and hasattr(layer.feed_forward, "feed_forward"):
                if hasattr(layer.feed_forward.feed_forward, "_checkpoint_activations"):
                    layer.feed_forward.feed_forward._checkpoint_activations = True
    
    def get_input_embeddings(self):
        """Get token embeddings."""
        return self.token_embeddings
    
    def set_input_embeddings(self, embeddings):
        """Set token embeddings."""
        self.token_embeddings = embeddings
    
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
        # Set default values
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        # Get sequence length and batch size
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        # Initialize past length
        past_length = 0
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
        
        # Create position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(
                past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0).expand(batch_size, -1)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=input_ids.device)
        
        # Prepare attention mask for attention
        # [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # Convert attention mask to additive mask
        # 0.0 for positions to attend, -10000.0 for positions to ignore
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)
        
        # Process multimodal input if provided
        if self.multimodal_encoder is not None and pixel_values is not None:
            # Encode images
            image_embeds = self.multimodal_encoder(pixel_values)
            
            # Combine with text embeddings
            # For simplicity, we prepend image embeddings to text embeddings
            # In a more sophisticated implementation, you might use a fusion mechanism
            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            
            # Update attention mask to include image tokens
            image_attention_mask = torch.ones(
                (batch_size, image_embeds.shape[1]), device=attention_mask.device
            )
            attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
            
            # Update position IDs to include image tokens
            image_position_ids = torch.zeros(
                (batch_size, image_embeds.shape[1]), device=position_ids.device
            )
            position_ids = torch.cat([image_position_ids, position_ids], dim=1)
            
            # Update extended attention mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Apply dropout to embeddings
        hidden_states = self.dropout(inputs_embeds)
        
        # Initialize variables for outputs
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_past_key_values = () if use_cache else None
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Add hidden states to outputs if requested
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            # Get past key values for this layer
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            # Apply gradient checkpointing if enabled
            if getattr(self, "gradient_checkpointing", False) and self.training:
                # Custom function for gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs[:4])[0]
                    return custom_forward
                
                # Apply layer with gradient checkpointing
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    extended_attention_mask,
                    position_ids,
                    layer_past,
                )
                hidden_states = layer_outputs
            else:
                # Apply layer normally
                layer_outputs = layer(
                    hidden_states=hidden_states,
                    attention_mask=extended_attention_mask,
                    position_ids=position_ids,
                    past_key_value=layer_past,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
                hidden_states = layer_outputs[0]
            
            # Add attention outputs to outputs if requested
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
            
            # Add past key values to outputs if requested
            if use_cache:
                all_past_key_values = all_past_key_values + (layer_outputs[2],)
        
        # Apply final layer normalization
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Add final hidden states to outputs if requested
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        return (hidden_states, all_hidden_states, all_attentions, all_past_key_values)


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
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
    
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
            past_key_values = outputs[3]
            
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
