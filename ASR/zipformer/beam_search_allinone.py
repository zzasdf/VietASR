# Copyright    2024  VietASR        (authors: Based on Xiaomi Corp.)
#
# Beam Search with Native ILME (Internal LM Estimation) Subtraction
# for All-in-One ASR
#
# Licensed under the Apache License, Version 2.0 (the "License")

"""
Beam search decoding methods with Native ILME subtraction.

This module provides beam search decoding that leverages the All-in-One
trained model's LM mode for accurate Internal LM subtraction.

Key difference from standard beam_search.py:
- modified_beam_search_ilme(): Uses model's LM mode (trained with encoder=0)
  for native ILME subtraction, improving generalization with external LM.
"""

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import k2
import sentencepiece as spm
import torch
from torch import nn

# Import shared utilities from beam_search.py
from beam_search import (
    Hypothesis,
    HypothesisList,
    get_hyps_shape,
    DecodingResults,
)


def modified_beam_search_ilme(
    model: nn.Module,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    beam: int = 4,
    temperature: float = 1.0,
    blank_penalty: float = 0.0,
    ilme_scale: float = 0.1,  # NEW: ILME subtraction scale
    return_timestamps: bool = False,
) -> Union[List[List[int]], DecodingResults]:
    """Beam search with Native ILME (Internal LM Estimation) subtraction.
    
    This function is designed for All-in-One trained models where the joiner
    has been trained with LM mode (encoder=0). The ILME is "native" because
    the model has learned to behave as a Language Model when encoder is zeroed.
    
    ILME subtraction: final_logits = transducer_logits - ilme_scale * lm_logits
    
    Args:
      model:
        The All-in-One transducer model with SigmoidAttentionJoiner.
      encoder_out:
        Output from the encoder. Its shape is (N, T, C).
      encoder_out_lens:
        A 1-D tensor of shape (N,), containing number of valid frames.
      beam:
        Number of active paths during the beam search.
      temperature:
        Softmax temperature.
      blank_penalty:
        Penalty for selecting blank token.
      ilme_scale:
        Scale for Internal LM subtraction. Default 0.1 (như paper).
        0.0 = no ILME subtraction (equivalent to standard beam search)
      return_timestamps:
        Whether to return timestamps.
        
    Returns:
      If return_timestamps is False, return the decoded result.
      Else, return a DecodingResults object.
    """
    assert encoder_out.ndim == 3, encoder_out.shape
    assert encoder_out.size(0) >= 1, encoder_out.size(0)

    packed_encoder_out = torch.nn.utils.rnn.pack_padded_sequence(
        input=encoder_out,
        lengths=encoder_out_lens.cpu(),
        batch_first=True,
        enforce_sorted=False,
    )

    blank_id = model.decoder.blank_id
    unk_id = getattr(model, "unk_id", blank_id)
    context_size = model.decoder.context_size
    device = next(model.parameters()).device

    batch_size_list = packed_encoder_out.batch_sizes.tolist()
    N = encoder_out.size(0)
    assert torch.all(encoder_out_lens > 0), encoder_out_lens
    assert N == batch_size_list[0], (N, batch_size_list)

    B = [HypothesisList() for _ in range(N)]
    for i in range(N):
        B[i].add(
            Hypothesis(
                ys=[-1] * (context_size - 1) + [blank_id],
                log_prob=torch.zeros(1, dtype=torch.float32, device=device),
                timestamp=[],
            )
        )

    # Project encoder output once
    encoder_out_proj = model.joiner.encoder_proj(packed_encoder_out.data)

    offset = 0
    finalized_B = []
    
    for t, batch_size in enumerate(batch_size_list):
        start = offset
        end = offset + batch_size
        current_encoder_out = encoder_out_proj[start:end]
        current_encoder_out = current_encoder_out.unsqueeze(1).unsqueeze(1)
        # Shape: (batch_size, 1, 1, joiner_dim)
        offset = end

        finalized_B = B[batch_size:] + finalized_B
        B = B[:batch_size]

        hyps_shape = get_hyps_shape(B).to(device)

        A = [list(b) for b in B]
        B = [HypothesisList() for _ in range(batch_size)]

        ys_log_probs = torch.cat(
            [hyp.log_prob.reshape(1, 1) for hyps in A for hyp in hyps]
        )  # (num_hyps, 1)

        decoder_input = torch.tensor(
            [hyp.ys[-context_size:] for hyps in A for hyp in hyps],
            device=device,
            dtype=torch.int64,
        )  # (num_hyps, context_size)

        decoder_out = model.decoder(decoder_input, need_pad=False).unsqueeze(1)
        decoder_out_proj = model.joiner.decoder_proj(decoder_out)
        # Shape: (num_hyps, 1, 1, joiner_dim)

        # Expand encoder output for all hypotheses
        current_encoder_out = torch.index_select(
            current_encoder_out,
            dim=0,
            index=hyps_shape.row_ids(1).to(torch.int64),
        )  # (num_hyps, 1, 1, joiner_dim)

        # Get transducer logits (normal mode)
        logits = model.joiner(
            current_encoder_out,
            decoder_out_proj,
            project_input=False,
            mode="transducer",  # Explicit mode
        )  # (num_hyps, 1, 1, vocab_size)

        # Native ILME subtraction
        if ilme_scale != 0:
            # LM logits: encoder zeroed (model trained with this mode)
            lm_logits = model.joiner(
                torch.zeros_like(current_encoder_out),  # Zero encoder
                decoder_out_proj,
                project_input=False,
                mode="lm",  # LM mode
            )
            # Subtract internal LM (như paper Eq. 24)
            logits = logits - ilme_scale * lm_logits

        logits = logits.squeeze(1).squeeze(1)  # (num_hyps, vocab_size)

        if blank_penalty != 0:
            logits[:, 0] -= blank_penalty

        log_probs = (logits / temperature).log_softmax(dim=-1)
        log_probs.add_(ys_log_probs)

        vocab_size = log_probs.size(-1)
        log_probs = log_probs.reshape(-1)

        row_splits = hyps_shape.row_splits(1) * vocab_size
        log_probs_shape = k2.ragged.create_ragged_shape2(
            row_splits=row_splits, cached_tot_size=log_probs.numel()
        )
        ragged_log_probs = k2.RaggedTensor(shape=log_probs_shape, value=log_probs)

        for i in range(batch_size):
            topk_log_probs, topk_indexes = ragged_log_probs[i].topk(beam)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                topk_hyp_indexes = (topk_indexes // vocab_size).tolist()
                topk_token_indexes = (topk_indexes % vocab_size).tolist()

            for k in range(len(topk_hyp_indexes)):
                hyp_idx = topk_hyp_indexes[k]
                hyp = A[i][hyp_idx]
                new_ys = hyp.ys[:]
                new_token = topk_token_indexes[k]
                new_timestamp = hyp.timestamp[:]
                
                if new_token not in (blank_id, unk_id):
                    new_ys.append(new_token)
                    new_timestamp.append(t)

                new_log_prob = topk_log_probs[k]

                new_hyp = Hypothesis(
                    ys=new_ys,
                    log_prob=new_log_prob,
                    timestamp=new_timestamp,
                )
                B[i].add(new_hyp)

    B = B + finalized_B

    best_hyps = [b.get_most_probable(length_norm=True) for b in B]

    sorted_ans = [h.ys[context_size:] for h in best_hyps]
    sorted_timestamps = [h.timestamp for h in best_hyps]
    
    ans = []
    ans_timestamps = []
    unsorted_indices = packed_encoder_out.unsorted_indices.tolist()
    for i in range(N):
        ans.append(sorted_ans[unsorted_indices[i]])
        ans_timestamps.append(sorted_timestamps[unsorted_indices[i]])

    if not return_timestamps:
        return ans
    else:
        return DecodingResults(
            hyps=ans,
            timestamps=ans_timestamps,
        )


# Export for convenience
__all__ = ["modified_beam_search_ilme"]
