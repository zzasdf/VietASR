# Copyright    2024  VietASR        (authors: Based on Xiaomi Corp.)
#
# All-in-One ASR Model with LM Mode Support
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

from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_interface import EncoderInterface
from icefall.utils import add_sos, make_pad_mask
from scaling import ScaledLinear


class AsrModelAllInOne(nn.Module):
    """All-in-One ASR Model với Sigmoid Attention Joiner và LM Mode.
    
    Differences from standard AsrModel:
    1. Joiner: SigmoidAttentionJoiner với multi-mode support
    2. forward_lm(): Train joiner như Language Model (encoder=0)
    3. Joint training: Transducer + CTC + LM loss
    
    Reference:
    - All-in-One ASR Paper
    """
    
    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        decoder: Optional[nn.Module] = None,
        joiner: Optional[nn.Module] = None,
        attention_decoder: Optional[nn.Module] = None,
        encoder_dim: int = 384,
        decoder_dim: int = 512,
        vocab_size: int = 500,
        use_transducer: bool = True,
        use_ctc: bool = False,
        use_attention_decoder: bool = False,
        use_lm_mode: bool = True,  # NEW: Enable LM mode training
    ):
        """A joint CTC & Transducer ASR model with LM mode support.

        Args:
          encoder_embed:
            It is a Convolutional 2D subsampling module.
          encoder:
            It is the transcription network in the paper.
          decoder:
            It is the prediction network in the paper.
          joiner:
            SigmoidAttentionJoiner với mode support ("transducer", "lm", "ctc")
          use_transducer:
            Whether use transducer head. Default: True.
          use_ctc:
            Whether use CTC head. Default: False.
          use_lm_mode:
            Whether train with LM mode loss. Default: True.
        """
        super().__init__()

        assert (
            use_transducer or use_ctc
        ), f"At least one of them should be True"

        assert isinstance(encoder, EncoderInterface), type(encoder)

        self.encoder_embed = encoder_embed
        self.encoder = encoder
        
        self.vocab_size = vocab_size
        self.use_lm_mode = use_lm_mode

        self.use_transducer = use_transducer
        if use_transducer:
            assert decoder is not None
            assert hasattr(decoder, "blank_id")
            assert joiner is not None

            self.decoder = decoder
            self.joiner = joiner

            self.simple_am_proj = ScaledLinear(
                encoder_dim, vocab_size, initial_scale=0.25
            )
            self.simple_lm_proj = ScaledLinear(
                decoder_dim, vocab_size, initial_scale=0.25
            )
        else:
            assert decoder is None
            assert joiner is None

        self.use_ctc = use_ctc
        if use_ctc:
            self.ctc_output = nn.Sequential(
                nn.Dropout(p=0.1),
                nn.Linear(encoder_dim, vocab_size),
                nn.LogSoftmax(dim=-1),
            )

        self.use_attention_decoder = use_attention_decoder
        if use_attention_decoder:
            self.attention_decoder = attention_decoder
        else:
            assert attention_decoder is None

    def forward_encoder(
        self, x: torch.Tensor, x_lens: torch.Tensor, final_downsample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder outputs."""
        x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)
        x = x.permute(1, 0, 2)

        encoder_out, encoder_out_lens = self.encoder(
            x, x_lens, src_key_padding_mask, final_downsample
        )

        encoder_out = encoder_out.permute(1, 0, 2)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

        return encoder_out, encoder_out_lens

    def forward_ctc(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CTC loss."""
        ctc_output = self.ctc_output(encoder_out)

        ctc_loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_output.permute(1, 0, 2),
            targets=targets.cpu(),
            input_lengths=encoder_out_lens.cpu(),
            target_lengths=target_lengths.cpu(),
            reduction="sum",
        )
        return ctc_loss

    def forward_transducer(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        y: k2.RaggedTensor,
        y_lens: torch.Tensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss."""
        blank_id = self.decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        decoder_out = self.decoder(sos_y_padded)

        y_padded = y.pad(mode="constant", padding_value=0)
        y_padded = y_padded.to(torch.int64)
        
        boundary = torch.zeros(
            (encoder_out.size(0), 4),
            dtype=torch.int64,
            device=encoder_out.device,
        )
        boundary[:, 2] = y_lens
        boundary[:, 3] = encoder_out_lens

        lm = self.simple_lm_proj(decoder_out)
        am = self.simple_am_proj(encoder_out)

        with torch.cuda.amp.autocast(enabled=False):
            simple_loss, (px_grad, py_grad) = k2.rnnt_loss_smoothed(
                lm=lm.float(),
                am=am.float(),
                symbols=y_padded,
                termination_symbol=blank_id,
                lm_only_scale=lm_scale,
                am_only_scale=am_scale,
                boundary=boundary,
                reduction="sum",
                return_grad=True,
            )

        ranges = k2.get_rnnt_prune_ranges(
            px_grad=px_grad,
            py_grad=py_grad,
            boundary=boundary,
            s_range=prune_range,
        )

        am_pruned, lm_pruned = k2.do_rnnt_pruning(
            am=self.joiner.encoder_proj(encoder_out),
            lm=self.joiner.decoder_proj(decoder_out),
            ranges=ranges,
        )

        # Use transducer mode for joiner
        logits = self.joiner(am_pruned, lm_pruned, project_input=False, mode="transducer")

        with torch.cuda.amp.autocast(enabled=False):
            pruned_loss = k2.rnnt_loss_pruned(
                logits=logits.float(),
                symbols=y_padded,
                ranges=ranges,
                termination_symbol=blank_id,
                boundary=boundary,
                reduction="sum",
            )

        return simple_loss, pruned_loss

    def forward_lm(
        self,
        decoder_out: torch.Tensor,
        y_padded: torch.Tensor,
        y_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LM loss for All-in-One training.
        
        Train joiner với mode="lm" (encoder zeroed) để học như Language Model.
        Điều này cho phép Native ILME subtraction khi decode.
        
        Args:
            decoder_out: Output từ decoder, shape (N, U, decoder_dim)
            y_padded: Target labels, shape (N, U)
            y_lens: Length of each target sequence, shape (N,)
            
        Returns:
            LM loss (scalar)
        """
        blank_id = self.decoder.blank_id
        
        # Project decoder output
        dec_proj = self.joiner.decoder_proj(decoder_out)
        
        # Create zero encoder (LM mode)
        enc_zero = torch.zeros_like(dec_proj)
        
        # Get LM logits (mode="lm" zeros encoder internally, but we also zero it here)
        lm_logits = self.joiner(enc_zero, dec_proj, project_input=False, mode="lm")
        # Shape: (N, U, vocab_size)
        
        # Shift for next-token prediction: predict y[1:] from decoder_out[:-1]
        # decoder_out[i] is based on y[0:i], should predict y[i]
        # So lm_logits[:, :-1] predicts y_padded[:, 1:]
        
        # Create mask cho valid positions
        batch_size, max_len = y_padded.shape
        mask = torch.arange(max_len, device=y_padded.device).unsqueeze(0) < y_lens.unsqueeze(1)
        
        # Flatten for cross entropy
        # lm_logits: (N, U, V) -> (N*U, V)
        # y_padded: (N, U) -> (N*U,)
        lm_logits_flat = lm_logits.view(-1, self.vocab_size)
        targets_flat = y_padded.view(-1)
        mask_flat = mask.view(-1)
        
        # Cross entropy với mask (ignore padding)
        lm_loss = F.cross_entropy(
            lm_logits_flat[mask_flat],
            targets_flat[mask_flat],
            ignore_index=blank_id,
            reduction="sum",
        )
        
        return lm_loss

    def forward(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
        prune_range: int = 5,
        am_scale: float = 0.0,
        lm_scale: float = 0.0,
        lm_loss_scale: float = 0.1,  # NEW: weight for LM loss
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
          x: Audio features, shape (N, T, C)
          x_lens: Feature lengths, shape (N,)
          y: Target labels (ragged tensor)
          lm_loss_scale: Weight for LM loss (default 0.1 như paper)
          
        Returns:
          (simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, lm_loss)
        """
        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

        device = x.device

        # Compute encoder outputs
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        # Transducer loss
        if self.use_transducer:
            simple_loss, pruned_loss = self.forward_transducer(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                y=y.to(device),
                y_lens=y_lens,
                prune_range=prune_range,
                am_scale=am_scale,
                lm_scale=lm_scale,
            )
        else:
            simple_loss = torch.empty(0)
            pruned_loss = torch.empty(0)

        # CTC loss
        if self.use_ctc:
            targets = y.values
            ctc_loss = self.forward_ctc(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                targets=targets,
                target_lengths=y_lens,
            )
        else:
            ctc_loss = torch.empty(0)

        # Attention decoder loss
        if self.use_attention_decoder:
            attention_decoder_loss = self.attention_decoder.calc_att_loss(
                encoder_out=encoder_out,
                encoder_out_lens=encoder_out_lens,
                ys=y.to(device),
                ys_lens=y_lens.to(device),
            )
        else:
            attention_decoder_loss = torch.empty(0)

        # LM loss (NEW for All-in-One)
        if self.use_lm_mode and self.use_transducer:
            blank_id = self.decoder.blank_id
            sos_y = add_sos(y.to(device), sos_id=blank_id)
            sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
            decoder_out = self.decoder(sos_y_padded)
            
            y_padded = y.to(device).pad(mode="constant", padding_value=0)
            
            lm_loss = self.forward_lm(
                decoder_out=decoder_out,
                y_padded=y_padded,
                y_lens=y_lens,
            )
        else:
            lm_loss = torch.empty(0)

        return simple_loss, pruned_loss, ctc_loss, attention_decoder_loss, lm_loss
