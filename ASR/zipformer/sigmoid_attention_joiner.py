# Copyright    2024  VietASR        (authors: Based on Xiaomi Corp.)
#
# Sigmoid Attention Joiner for All-in-One ASR
# Based on paper: "All-in-One ASR: Multi-Mode Joiner for Unified Transducer, AED, CTC, and LM"
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

import torch
import torch.nn as nn
from scaling import ScaledLinear


class SigmoidAttentionJoiner(nn.Module):
    """Joiner với Sigmoid Attention cho All-in-One ASR.
    
    Key features:
    - Sigmoid Attention: Cho phép LM mode (encoder=0) hoạt động ổn định
    - Standard Projection: Output raw logits, k2-compatible
    - Multi-head attention: Tương thích với paper
    
    Modes:
    - "transducer": Normal (encoder + decoder with sigmoid attention)
    - "lm": LM mode (encoder=0, cho ILME training)
    - "ctc": CTC mode (decoder=0, attention=0.5)
    
    Reference:
    - All-in-One ASR Paper Eq. (13): α = Sigmoid(k·q / √d)
    """
    
    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        joiner_dim: int,
        vocab_size: int,
        num_heads: int = 8,
    ):
        """
        Args:
            encoder_dim: Dimension of encoder output
            decoder_dim: Dimension of decoder output
            joiner_dim: Internal dimension of joiner
            vocab_size: Number of output tokens (including blank)
            num_heads: Number of attention heads
        """
        super().__init__()
        
        assert joiner_dim % num_heads == 0, \
            f"joiner_dim ({joiner_dim}) must be divisible by num_heads ({num_heads})"
        
        # Standard projections (giữ nguyên pattern từ original Joiner)
        self.encoder_proj = ScaledLinear(encoder_dim, joiner_dim, initial_scale=0.25)
        self.decoder_proj = ScaledLinear(decoder_dim, joiner_dim, initial_scale=0.25)
        
        # Sigmoid Attention components
        self.num_heads = num_heads
        self.head_dim = joiner_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # LayerNorm cho Query và Key/Value (như paper Eq. 10-12)
        self.ln_query = nn.LayerNorm(joiner_dim)
        self.ln_kv = nn.LayerNorm(joiner_dim)
        
        # Q, K, V projections
        self.query = nn.Linear(joiner_dim, joiner_dim)
        self.key = nn.Linear(joiner_dim, joiner_dim)
        self.value = nn.Linear(joiner_dim, joiner_dim)
        
        # Output projection
        self.out_proj = nn.Linear(joiner_dim, joiner_dim)
        
        # Standard output layer (RAW LOGITS - k2 compatible!)
        self.output_linear = nn.Linear(joiner_dim, vocab_size)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize attention parameters."""
        # Xavier initialization for attention
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        # Zero bias
        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
        if self.key.bias is not None:
            nn.init.zeros_(self.key.bias)
        if self.value.bias is not None:
            nn.init.zeros_(self.value.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)
    
    def _sigmoid_attention(
        self,
        enc: torch.Tensor,
        dec: torch.Tensor,
    ) -> torch.Tensor:
        """Framewise sigmoid attention.
        
        Paper Eq. (13-15):
        α = Sigmoid(k·q / √d)   # Framewise, không cần full context
        c = α × v
        
        Args:
            enc: Projected encoder output, shape (..., joiner_dim)
            dec: Projected decoder output, shape (..., joiner_dim)
            
        Returns:
            Context vector, shape (..., joiner_dim)
        """
        # Query từ decoder, Key/Value từ encoder
        q = self.query(self.ln_query(dec))  # (..., joiner_dim)
        k = self.key(self.ln_kv(enc))
        v = self.value(self.ln_kv(enc))
        
        # Save original shape for reshaping back
        original_shape = q.shape[:-1]  # (...,)
        
        # Reshape for multi-head: (..., num_heads, head_dim)
        q = q.view(*original_shape, self.num_heads, self.head_dim)
        k = k.view(*original_shape, self.num_heads, self.head_dim)
        v = v.view(*original_shape, self.num_heads, self.head_dim)
        
        # Sigmoid attention (framewise)
        # Khi enc=0: k=0 → q·k=0 → sigmoid(0)=0.5 → stable
        attn = torch.sigmoid((q * k).sum(dim=-1, keepdim=True) * self.scale)
        context = attn * v  # (..., num_heads, head_dim)
        
        # Reshape back
        context = context.view(*original_shape, -1)  # (..., joiner_dim)
        context = self.out_proj(context)
        
        return context
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_out: torch.Tensor,
        project_input: bool = True,
        mode: str = "transducer",
    ) -> torch.Tensor:
        """
        Args:
            encoder_out: Output from encoder, shape (N, T, s_range, C) hoặc (N, 1, 1, C)
            decoder_out: Output from decoder, shape (N, T, s_range, C) hoặc (N, 1, 1, C)
            project_input: Whether to apply input projections
            mode: "transducer", "lm", hoặc "ctc"
            
        Returns:
            Raw logits, shape (..., vocab_size) - k2 compatible
        """
        assert encoder_out.ndim == decoder_out.ndim, (
            encoder_out.shape,
            decoder_out.shape,
        )
        
        if project_input:
            enc = self.encoder_proj(encoder_out)
            dec = self.decoder_proj(decoder_out)
        else:
            enc = encoder_out
            dec = decoder_out
        
        if mode == "lm":
            # LM mode: Zero encoder để train như Language Model
            # sigmoid(0) = 0.5 → context = 0.5 * value (stable)
            enc = torch.zeros_like(enc)
        elif mode == "ctc":
            # CTC mode: Zero decoder, attention weight = 0.5
            dec = torch.zeros_like(dec)
        
        # Sigmoid attention
        context = self._sigmoid_attention(enc, dec)
        
        # Combine: h = tanh(dec + context)
        # Như paper Eq. (16): h_joiner = tanh(h_pred + c)
        h = torch.tanh(dec + context)
        
        # Output: RAW LOGITS (k2 sẽ tự apply log-softmax)
        logits = self.output_linear(h)
        
        return logits


# Backward compatibility: alias cho import cũ
Joiner = SigmoidAttentionJoiner
