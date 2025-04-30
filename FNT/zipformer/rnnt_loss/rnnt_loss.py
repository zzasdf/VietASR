# -*- coding: utf-8 -*-

import io
import os
import logging
import math
import warnings
from typing import Optional, Tuple, Union
import torch
from torch import Tensor
from torch.utils.cpp_extension import load

logger = logging.getLogger(__name__)

build_dir = os.path.dirname(os.path.abspath(__file__)) + "/rnnt_loss_cpp"
os.makedirs(build_dir, exist_ok=True)
logger.info("Compiling C++ code to: {}".format(build_dir))
# logging.info(os.environ["INCLUDEPATH"])
if "INCLUDEPATH" not in os.environ:
    os.environ["INCLUDEPATH"] = ''
transducer_loss = load(
    name='transducer_loss',
    extra_include_paths=[os.environ["INCLUDEPATH"] + ":/usr/local/cuda/include"],
    sources=[os.path.join(os.path.dirname(__file__), source_file) for source_file in ["rnnt_loss.cpp", "gpu/compute.cu"]],
    build_directory=build_dir,
    verbose=True
)
logger.info("successfully build rnnt_loss")
with open(build_dir + '/complete', 'w') as f:
    f.write("DONE")

def rnnt_loss(
    logits: Tensor,
    targets: Tensor,
    logit_lengths: Tensor,
    target_lengths: Tensor,
    blank: int = -1,
    clamp: float = -1,
    reduction: str = "mean",
):
    """Compute the RNN Transducer loss from *Sequence Transduction with Recurrent Neural Networks*
    [:footcite:`graves2012sequence`].

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    The RNN Transducer loss extends the CTC loss by defining a distribution over output
    sequences of all lengths, and by jointly modelling both input-output and output-output
    dependencies.

    Args:
        logits (Tensor): Tensor of dimension `(batch, max seq length, max target length + 1, class)`
            containing output from joiner
        targets (Tensor): Tensor of dimension `(batch, max target length)` containing targets with zero padded
        logit_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of each sequence from encoder
        target_lengths (Tensor): Tensor of dimension `(batch)` containing lengths of targets for each sequence
        blank (int, optional): blank label (Default: ``-1``)
        clamp (float, optional): clamp for gradients (Default: ``-1``)
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. (Default: ``'mean'``)
    Returns:
        Tensor: Loss with the reduction option applied. If ``reduction`` is  ``'none'``, then size `(batch)`,
        otherwise scalar.
    """
    if reduction not in ["none", "mean", "sum"]:
        raise ValueError("reduction should be one of 'none', 'mean', or 'sum'")

    if blank < 0:  # reinterpret blank index if blank < 0.
        blank = logits.shape[-1] + blank

    # new_logits = logits.clone()
    costs, _ = transducer_loss.rnnt_loss(
        logits,
        targets,
        logit_lengths,
        target_lengths,
        blank,
        clamp,
    )

    if reduction == "mean":
        return costs.mean()
    elif reduction == "sum":
        return costs.sum()

    return costs


