# Copyright    2021-2024  Xiaomi Corp.        (authors: Fangjun Kuang,
#                                                       Wei Kang,
#                                                       Zengwei Yao,
#                                                       Yifan Yang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
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
import contextlib
from typing import Optional, Tuple

import k2
import torch
import torch.nn as nn
from utils import GradMultiply
from scaling import ScaledLinear

from icefall.utils import add_sos


class HubertFNTAsrModel(nn.Module):
	def __init__(
		self,
		encoder,
		blank_decoder: nn.Module = None,
		vocab_decoder: nn.Module = None,
		joiner: nn.Module = None,
		encoder_feature_layer: int = -1,
	):
		"""
		Args:
		  encoder:
			Hubert Encoder. Different from factorized model,
            Hubert encoder has both encoder_embed and zipformer2
            Its accepts
			inputs: `x` of (N, T, encoder_dim).
			It returns two tensors: `logits` of shape (N, T, encoder_dim) and
			`logit_lens` of shape (N,).
		  decoder:
			It is the prediction network in the paper. Its input shape
			is (N, U) and its output shape is (N, U, decoder_dim).
			It should contain one attribute: `blank_id`.
		  joiner:
			It has two inputs with shapes: (N, T, encoder_dim) and (N, U, decoder_dim).
			Its output shape is (N, T, U, vocab_size). Note that its output contains
			unnormalized probs, i.e., not processed by log-softmax.
		"""
		super().__init__()

		self.encoder = encoder
		self.encoder_feature_layer = encoder_feature_layer

        assert blank_decoder is not None
        assert vocab_decoder is not None
        assert hasattr(blank_decoder, "blank_id")
        assert joiner is not None

        self.blank_decoder = blank_decoder
        self.vocab_decoder = vocab_decoder
        self.joiner = joiner


	def forward_encoder(
		self,
		x: torch.Tensor,
		padding_mask: Optional[torch.Tensor] = None,
		do_final_down_sample: bool = True
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute encoder outputs.
		Args:
		  x:
			A 2-D tensor of shape (N, T).

		Returns:
		  encoder_out:
			Encoder output, of shape (N, T, C).
		  encoder_out_lens:
			Encoder output lengths, of shape (N,).
		"""
		if padding_mask is None:
			padding_mask = torch.zeros_like(x, dtype=torch.bool)

		encoder_out, padding_mask, encoder_out_lens = self.encoder.extract_features(
			source=x,
			padding_mask=padding_mask,
			mask=self.encoder.training,
			output_layer = self.encoder_feature_layer,
			do_final_down_sample = do_final_down_sample
		)
		# encoder_out_lens = torch.sum(~padding_mask, dim=1)
		assert torch.all(encoder_out_lens > 0), encoder_out_lens

		return encoder_out, encoder_out_lens


	def forward_transducer(
		self,
		encoder_out: torch.Tensor,
		encoder_out_lens: torch.Tensor,
		y: k2.RaggedTensor,
		y_lens: torch.Tensor,
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute Transducer loss.
		Args:
		  encoder_out:
			Encoder output, of shape (N, T, C).
		  encoder_out_lens:
			Encoder output lengths, of shape (N,).
		  y:
			A ragged tensor with 2 axes [utt][label]. It contains labels of each
			utterance.
		"""
		blank_id = self.blank_decoder.blank_id
		# sos_id = self.vocab_decoder.sos_id
		sos_y = add_sos(y, sos_id=blank_id)

		lm_target = torch.clone(y.values)
		lm_target = lm_target.to(torch.int64)
		lm_target[lm_target>blank_id] -= 1
		# shift because the lm_predictor output does not include the blank_id

		# sos_y_padded: [B, S + 1], start with SOS.
		sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
		sos_y_padded = sos_y_padded.to(torch.int64)

		# decoder_out: [B, S + 1, decoder_dim]
		blank_decoder_out, _ = self.blank_decoder(sos_y_padded)
		vocab_decoder_out, _ = self.vocab_decoder(sos_y_padded)

		lm_lprobs = self.joiner.vocab_lm_probs_no_softmax(vocab_decoder_out)		
		# with no log_softmax, lm_lprobs should not do log_softmax before reshape

		logits, _ = self.joiner(encoder_out, blank_decoder_out, vocab_decoder_out)

		lm_lprobs = [torch.nn.functional.log_softmax(item[:y_len], dim=-1) for item, y_len in zip(lm_lprobs, y_lens)] 
		"""
		sos_y is padded with <sos> in the begining, lm_lprobs will also be S+1 long,
		but we just clip it to y_len, which will be of same length as the original y.
		As for lm_target, it does not do any padding, no blank_id either.
		Therefore, ignore_index is not applicable.
		Meanwhile, nll_loss require input to be log_softmax
		"""
		lm_lprobs = torch.cat(lm_lprobs)

		lm_loss = torch.nn.functional.nll_loss(
			lm_lprobs,
			lm_target,
			reduction="sum"
		)

		# Note: y does not start with SOS
		# y_padded : [B, S]
		y_padded = y.pad(mode="constant", padding_value=0)
		y_padded = y_padded.to(torch.int64)
		boundary = torch.zeros(
			(encoder_out.size(0), 4),
			dtype=torch.int64,
			device=encoder_out.device,
		)
		boundary[:, 2] = y_lens
		boundary[:, 3] = encoder_out_lens

		try:
			from k2 import rnnt_loss
		except ImportError:
			raise ImportError("cannot import rnnt_loss from k2")

		rnnt_loss = rnnt_loss(
			logits=logits,
			symbols=y_padded,
			termination_symbol=blank_id,
			boundary=boundary,
			reduction="sum",
		)
	
		return rnnt_loss, lm_loss


	def forward(
		self,
		x: torch.Tensor,
		y: k2.RaggedTensor,
		padding_mask: Optional[torch.Tensor] = None,
		freeze_encoder: bool = False,
		encoder_grad_scale: float = 1,
	) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Args:
		  x:
			A 2-D tensor of shape (N, T).
		  y:
			A ragged tensor with 2 axes [utt][label]. It contains labels of each
			utterance.
		Returns:
		"""
		assert x.ndim == 3, x.shape
		assert y.num_axes == 2, y.num_axes

		assert x.size(0) == y.dim0, (x.shape, y.dim0)

		# Compute encoder outputs
		if freeze_encoder:
			with torch.no_grad():
				encoder_out, encoder_out_lens = self.forward_encoder(x, padding_mask)
		else:
			encoder_out, encoder_out_lens = self.forward_encoder(x, padding_mask)
			if encoder_grad_scale!=1:
				GradMultiply.apply(encoder_out, encoder_grad_scale)

		row_splits = y.shape.row_splits(1)
		y_lens = row_splits[1:] - row_splits[:-1]

        rnnt_loss, lm_loss = self.forward_transducer(
            encoder_out=encoder_out,
            encoder_out_lens=encoder_out_lens,
            y=y.to(x.device),
            y_lens=y_lens,
        )

		return rnnt_loss, lm_loss
