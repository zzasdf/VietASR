# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
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

"""
Note we use `rnnt_loss` from torchaudio, which exists only in
torchaudio >= v0.10.0. It also means you have to use torch >= v1.10.0
"""
import k2
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional
from encoder_interface import EncoderInterface
from subsampling import Conv2dSubsampling
from icefall.utils import add_sos, make_pad_mask
from zipformer import Zipformer2


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
  return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


class FactorizeTransducer(nn.Module):
    """It implements https://arxiv.org/pdf/1211.3711.pdf
    "Sequence Transduction with Recurrent Neural Networks"
    """

    def __init__(
        self,
        encoder_embed: nn.Module,
        encoder: EncoderInterface,
        blank_decoder: nn.Module,
        vocab_decoder: nn.Module,
        joiner: nn.Module,
		args,
    ):
        """
        Args:
          encoder:
            It is the transcription network in the paper. Its accepts
            two inputs: `x` of (N, T, C) and `x_lens` of shape (N,).
            It returns two tensors: `logits` of shape (N, T, C) and
            `logit_lens` of shape (N,).
          decoder:
            It is the prediction network in the paper. Its input shape
            is (N, U) and its output shape is (N, U, C). It should contain
            one attribute: `blank_id`.
          joiner:
            It has two inputs with shapes: (N, T, C) and (N, U, C). Its
            output shape is (N, T, U, C). Note that its output contains
            unnormalized probs, i.e., not processed by log-softmax.
        """
        super().__init__()
        # assert isinstance(encoder, EncoderInterface)
        self.encoder_embed = encoder_embed
        self.encoder = encoder
        self.blank_decoder = blank_decoder		# blank decoder and vocab decoder are basically the same structure
        self.vocab_decoder = vocab_decoder
        self.joiner = joiner
		self.train_stage = args.train_stage

		"""
		Blank decoder and vocab decoder are basically the same structure,
		output with same decoder dim, the differences lies in the joiner
		With embedding layer, num_embeddings equal the vocab_size (with <blank> token)
		however, the vocab does not include a <pad> token, and uses <blk> as padding_idx
		"""


    @classmethod
	def build_encoder_embed(params) -> nn.Module:
		"""
		encoder_embed converts the input of shape (N, T, num_features)
		to the shape (N, (T - 7) // 2, encoder_dims).
		That is, it does two things simultaneously:
		  (1) subsampling: T -> (T - 7) // 2
		  (2) embedding: num_features -> encoder_dims
		In the normal configuration, we will downsample once more at the end
		by a factor of 2, and most of the encoder stacks will run at a lower
		sampling rate.
		"""
		
		encoder_embed = Conv2dSubsampling(
			in_channels=params.feature_dim,
			out_channels=_to_int_tuple(params.encoder_dim)[0],
			dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
		)
		if params.pretrain_path is not None:
			checkpoint = torch.load(params.pretrain_path, map_location=torch.device("cpu"))
			checkpoint = checkpoint['model']
			new_checkpoint = OrderedDict()
			prefix = "encoder_embed."
			for item in checkpoint:
				if item.startswith(prefix):
					new_checkpoint[item[len(prefix):]] = checkpoint[item]
			encoder_embed.load_state_dict(new_checkpoint)

		return encoder_embed


	@classmethod
	def build_encoder(params) -> nn.Module:
		encoder = Zipformer2(
			output_downsampling_factor=2,
			downsampling_factor=_to_int_tuple(params.downsampling_factor),
			num_encoder_layers=_to_int_tuple(params.num_encoder_layers),
			encoder_dim=_to_int_tuple(params.encoder_dim),
			encoder_unmasked_dim=_to_int_tuple(params.encoder_unmasked_dim),
			query_head_dim=_to_int_tuple(params.query_head_dim),
			pos_head_dim=_to_int_tuple(params.pos_head_dim),
			value_head_dim=_to_int_tuple(params.value_head_dim),
			pos_dim=params.pos_dim,
			num_heads=_to_int_tuple(params.num_heads),
			feedforward_dim=_to_int_tuple(params.feedforward_dim),
			cnn_module_kernel=_to_int_tuple(params.cnn_module_kernel),
			dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
			warmup_batches=4000.0,
			causal=params.causal,
			chunk_size=_to_int_tuple(params.chunk_size),
			left_context_frames=_to_int_tuple(params.left_context_frames),
		)
		if params.pretrain_path is not None:
			checkpoint = torch.load(params.pretrain_path, map_location=torch.device("cpu"))
			checkpoint = checkpoint['model']
			new_checkpoint = OrderedDict()
			prefix = "encoder."
			for item in checkpoint:
				if params.pretrain_type=="SSL":
					if item == "encoder.downsample_output.bias":
						continue
				if item.startswith(prefix):
					new_checkpoint[item[len(prefix):]] = checkpoint[item]
			missing_keys, unexpected_keys = encoder.load_state_dict(new_checkpoint, strict=False)
			print(f"missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}")
		return encoder


	@classmethod
	def build_decoder(params) -> nn.Module:
		decoder = FNTDecoder(
			vocab_size=params.vocab_size,
			embedding_dim=params.decoder_embedding_dim,
			blank_id=params.blank_id,
			num_layers=params.num_decoder_layers,
			hidden_dim=params.decoder_dim,
			output_dim=-1 # we move the output project layer to the joiner)
		)
		return decoder


	@classmethod
	def build_joiner(params) -> nn.Module:
		joiner = FNTJoiner(
			joint_dim=params.joiner_dim,
            encoder_hidden_dim=max(_to_int_tuple(params.encoder_dim)),
            decoder_hidden_dim=params.decoder_dim,
            vocab_size=params.vocab_size,
            blank_id=params.blank_id
		)
		return joiner


	@classmethod
	def build_model(params) -> nn.Module:
		assert params.use_transducer or params.use_ctc, (
			f"At least one of them should be True, "
			f"but got params.use_transducer={params.use_transducer}, "
			f"params.use_ctc={params.use_ctc}"
		)
		assert params.model_type == "FNT", (
			f"Invalid model type: {params.model_type}, "
			f"model type should only be FNT"
		)

		encoder_embed = build_encoder_embed(params)
		encoder = build_encoder(params)

		if params.use_transducer:
			blank_decoder = build_decoder(params)
			vocab_decoder = build_decoder(params)		# same structure for both blank and vocab decoder
			joiner = build_joiner(params)
		else:
			decoder = None
			joiner = None

		model = FactorizeTransducer(
			encoder_embed=encoder_embed,
			encoder=encoder,
			blank_decoder=blank_decoder,
			vocab_decoder=vocab_decoder,
			joiner=joiner,
			params,	
		)
		
		return model


	def forward_encoder(
		self, x: torch.Tensor, x_lens: torch.Tensor, final_downsample: bool = True
	) -> Tuple[torch.Tensor, torch.Tensor]:
		"""Compute encoder outputs.
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.

        Returns:
          encoder_out:
            Encoder output, of shape (N, T, C).
          encoder_out_lens:
            Encoder output lengths, of shape (N,).
        """
		x, x_lens = self.encoder_embed(x, x_lens)
        src_key_padding_mask = make_pad_mask(x_lens)

        # we use zipformer when the encoder_embed is not None, the input of zipformer is (T, N, C)
        x = x.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)

        encoder_out, encoder_out_lens = self.encoder(x, x_lens, src_key_padding_mask, final_downsample)		# added final_downsample

        encoder_out = encoder_out.permute(1, 0, 2)  # (T, N, C) ->(N, T, C)
        assert torch.all(encoder_out_lens > 0), (x_lens, encoder_out_lens)

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
            utterance.	[B, S]
		  y_lens: [B]
        """
		blank_id = self.blank_decoder.blank_id
        sos_y = add_sos(y, sos_id=blank_id)			# add blank_id at every sentence's beginning

        lm_target = torch.clone(y.values)			# changed to 1-dim (y could have different lengths), shape shift to [L]
        lm_target = lm_target.to(torch.int64)		
        lm_target[lm_target>blank_id] -= 1			# elements larger than blank_id will be shifted
		"""
		since we don't use a pad_id, we just concat the labels, which can be done by using its values
        shift because the lm_predictor output does not include the blank_id
		"""
		
		# sos_y_padded: [B, S + 1], start with SOS.
        sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
        sos_y_padded = sos_y_padded.to(torch.int64)

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

        # rnnt_loss requires 0 padded targets
        # Note: y does not start with SOS
        y_padded = y.pad(mode="constant", padding_value=0)

        assert hasattr(torchaudio.functional, "rnnt_loss"), (
            f"Current torchaudio version: {torchaudio.__version__}\n"
            "Please install a version >= 0.10.0"
        )

        rnnt_loss = torchaudio.functional.rnnt_loss(
            logits=logits,
            targets=y_padded,
            logit_lengths=x_lens,
            target_lengths=y_lens,
            blank=blank_id,
            reduction="sum",
        )
	
		return rnnt_loss, lm_loss

		

    def forward(
        self,
        x: torch.Tensor = None,
        x_lens: torch.Tensor,
        y: k2.RaggedTensor,
    ) -> torch.Tensor:
        """
        Args:
          x:
            A 3-D tensor of shape (N, T, C).
          x_lens:
            A 1-D tensor of shape (N,). It contains the number of frames in `x`
            before padding.
          y:
            A ragged tensor with 2 axes [utt][label]. It contains labels of each
            utterance.
        Returns:
          Return the transducer loss.
        """

        if self.train_stage == "adapt":
			#TODO: need further consideration on this part
            assert y.num_axes == 2, y.num_axes
            blank_id = self.vocab_decoder.blank_id
            row_splits = y.shape.row_splits(1)
            y_lens = row_splits[1:] - row_splits[:-1]

            lm_target = torch.clone(y.values)
            # since we don't use a pad_id, we just concat the labels, which can be done by using its values
            lm_target = lm_target.to(torch.int64)
            lm_target[lm_target > blank_id] -= 1
            # shift because the lm_predictor output does not include the blank_id

            sos_y = add_sos(y, sos_id=blank_id)

            sos_y_padded = sos_y.pad(mode="constant", padding_value=blank_id)
            sos_y_padded = sos_y_padded.to(torch.int64)
            decoder_out,_ = self.vocab_decoder(sos_y_padded)
            decoder_out = self.joiner.laynorm_proj_vocab_decoder(decoder_out)
            decoder_out = self.joiner.fc_out_decoder_vocab(decoder_out)
            lm_lprobs = torch.nn.functional.log_softmax(
                decoder_out,
                dim=-1,
            )
            lm_lprobs = [item[:y_len] for item, y_len in zip(lm_lprobs, y_lens)] 
            # note that the last output of each sentence is not used here
            lm_lprobs = torch.cat(lm_lprobs)
            lm_loss = torch.nn.functional.nll_loss(
                lm_lprobs,
                lm_target,
                reduction="sum")
            return lm_loss, y_lens

        assert x.ndim == 3, x.shape
        assert x_lens.ndim == 1, x_lens.shape
        assert y.num_axes == 2, y.num_axes

        assert x.size(0) == x_lens.size(0) == y.dim0

		device = x.device
		
		# compute output for encoder
        encoder_out, encoder_out_lens = self.forward_encoder(x, x_lens)

        # Now for the decoder, i.e., the prediction network
        row_splits = y.shape.row_splits(1)
        y_lens = row_splits[1:] - row_splits[:-1]

        rnnt_loss, lm_loss = self.forward_transducer(
			encoder_out=encoder_out,
			encoder_out_lens=encoder_out_lens,
			y=y.to(x.device),
			y_lens=y_lens,
		)

        return rnnt_loss, lm_loss


class FNTDecoder(nn.Module):
	def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        blank_id: int,
        num_layers: int,
        hidden_dim: int,
        output_dim: int,
        embedding_dropout: float = 0.0,
        rnn_dropout: float = 0.0,
    ):
        """
        Args:
          vocab_size:
            Number of tokens of the modeling unit including blank.
          embedding_dim:
            Dimension of the input embedding.
          blank_id:
            The ID of the blank symbol.
          num_layers:
            Number of LSTM layers.
          hidden_dim:
            Hidden dimension of LSTM layers.
          output_dim:
            Output dimension of the decoder.
          embedding_dropout:
            Dropout rate for the embedding layer.
          rnn_dropout:
            Dropout for LSTM layers.
        """
        super().__init__()
		self.vocab_size = vocab_size		# same for both blank and vocab decoder.
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=blank_id,			# should be dictionary without <blk> token, beware
        )
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.blank_id = blank_id
        if output_dim > 0:
          self.output_linear = nn.Linear(hidden_dim, output_dim)
        else:
          self.output_linear = lambda x:x


    def forward(
        self,
        y: torch.Tensor,
        states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
          y:
            A 2-D tensor of shape (N, U) with BOS prepended.
          states:
            A tuple of two tensors containing the states information of
            LSTM layers in this decoder.
        Returns:
          Return a tuple containing:

            - rnn_output, a tensor of shape (N, U, C)
            - (h, c), containing the state information for LSTM layers.
              Both are of shape (num_layers, N, C)
        """
        embedding_out = self.embedding(y)
        embedding_out = self.embedding_dropout(embedding_out)
        rnn_out, (h, c) = self.rnn(embedding_out, states)
        # out = self.output_linear(rnn_out)

        return rnn_out, (h, c)


class FNTJoiner(nn.Module):
    def __init__(self, joint_dim: int, encoder_hidden_dim:int, decoder_hidden_dim, vocab_size: int, blank_id: int):
        super().__init__()
        self.joint_dim = joint_dim
        self.encoder_embed_dim = encoder_hidden_dim
        self.decoder_embed_dim = decoder_hidden_dim
        self.blank_id = blank_id
        # add blank symbol in output layer
        self.out_dim = vocab_size
        self.out_blank_dim = 1
        self.out_vocab_dim = self.out_dim - self.out_blank_dim		# V - 1

        self.proj_encoder = nn.Linear(self.encoder_embed_dim, self.joint_dim)
        self.laynorm_proj_encoder = LayerNorm(self.joint_dim)
        self.proj_blank_decoder = nn.Linear(self.decoder_embed_dim, self.joint_dim)
        self.laynorm_proj_blank_decoder = LayerNorm(joint_dim)
        self.laynorm_proj_vocab_decoder = LayerNorm(decoder_hidden_dim)

        self.fc_out_blank = nn.Linear(self.joint_dim, self.out_blank_dim)
        self.fc_out_encoder_vocab = nn.Linear(self.joint_dim, self.out_vocab_dim)
        self.fc_out_decoder_vocab = nn.Linear(self.joint_dim, self.out_vocab_dim)		# V-1

        nn.init.normal_(self.proj_encoder.weight, mean=0, std=self.joint_dim**-0.5)
        nn.init.normal_(self.proj_blank_decoder.weight, mean=0, std=self.joint_dim**-0.5)
        nn.init.normal_(self.fc_out_blank.weight, mean=0, std=self.joint_dim**-0.5)
        nn.init.normal_(self.fc_out_encoder_vocab.weight, mean=0, std=self.joint_dim**-0.5)
        nn.init.normal_(self.fc_out_decoder_vocab.weight, mean=0, std=self.joint_dim**-0.5)

    # encoder_out: B x T x C
    # decoder_out: B X U x C

	def vocab_lm_probs_no_softmax(self, vocab_decoder_out):
		"""
		Args:
			vocab_decoder_out:
				Output from the vocab_decoder. Its shape is (N, U, C).
		Returns:
			output from fc_out_decoder_vocab, without doing log_softmax
			this will be used for lm_loss computation, as padded parts will be excluded.
			return tensor shape: (N, U, C)
		"""
		vocab_decoder_out = self.laynorm_proj_vocab_decoder(vocab_decoder_out)
        out_vocab_decoder = self.fc_out_decoder_vocab(vocab_decoder_out)
		return out_vocab_decoder


    def forward(self, encoder_out, blank_decoder_out, vocab_decoder_out):
        """
        Args:
          encoder_out:
            Output from the encoder. Its shape is (N, T, C).
          vocab_decoder_out:
            Output from the vocab_decoder. Its shape is (N, U, C).
          blank_decoder_out:
            Output from the blank_decoder. Its shape is (N, U, C).
        Returns:
          Return a tensor of shape (N, T, U, C).
        """
        encoder_out = self.laynorm_proj_encoder(self.proj_encoder(encoder_out))
        blank_decoder_out = self.laynorm_proj_blank_decoder(self.proj_blank_decoder(blank_decoder_out))
        vocab_decoder_out = self.laynorm_proj_vocab_decoder(vocab_decoder_out)

        out_blank = nn.functional.relu(encoder_out.unsqueeze(2) + blank_decoder_out.unsqueeze(1))
        out_blank = self.fc_out_blank(out_blank)

        out_vocab_encoder = self.fc_out_encoder_vocab(encoder_out)
        out_vocab_decoder = self.fc_out_decoder_vocab(vocab_decoder_out)
        out_vocab_decoder =  torch.nn.functional.log_softmax(out_vocab_decoder, dim=-1)		# lm_lprobs
        out_vocab = out_vocab_encoder.unsqueeze(2) + out_vocab_decoder.unsqueeze(1)
        out = torch.cat((out_vocab[:,:,:,:self.blank_id], out_blank, out_vocab[:,:,:,self.blank_id:]), dim=-1)
        return out, out_vocab_decoder
