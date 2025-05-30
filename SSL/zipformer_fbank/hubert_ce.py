# Copyright (c) Facebook, Inc. and its affiliates.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import math
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from scaling import ScheduledFloat
from subsampling import Conv2dSubsampling
from utils import LayerNorm
from zipformer import Zipformer2


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length)
            + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length)
                    + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length)
                + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError(f"this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask


def _to_int_tuple(s: str):
    return tuple(map(int, s.split(",")))


class HubertModel(nn.Module):
    def __init__(
        self,
        cfg,
    ) -> None:
        super().__init__()
        self.embed = _to_int_tuple(cfg.encoder_dim)[0]

        self.encoder_embed = Conv2dSubsampling(
            in_channels=cfg.feature_dim,
            out_channels=_to_int_tuple(cfg.encoder_dim)[0],
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
        )
        self.feature_ds_rate = 2 # TODO: this is from Conv2dSubsampling, I'm not sure if this is right
        self.feat2tar_ratio = (
            cfg.label_rate * self.feature_ds_rate / cfg.sample_rate
        )  # TODO feature_ds_rate 320
        encoder_input_dim = _to_int_tuple(cfg.encoder_dim)[0]
        encoder_output_dim = max(_to_int_tuple(cfg.encoder_dim))

        self.mask_before_cnn = cfg.mask_before_cnn
        self.mask_prob = cfg.mask_prob
        self.mask_selection = cfg.mask_selection
        self.mask_other = cfg.mask_other
        self.mask_length = cfg.mask_length
        self.no_mask_overlap = cfg.no_mask_overlap
        self.mask_min_space = cfg.mask_min_space

        self.mask_channel_prob = cfg.mask_channel_prob
        self.mask_channel_selection = cfg.mask_channel_selection
        self.mask_channel_other = cfg.mask_channel_other
        self.mask_channel_length = cfg.mask_channel_length
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap
        self.mask_channel_min_space = cfg.mask_channel_min_space

        self.dropout_input = nn.Dropout(cfg.dropout_input)
        self.dropout_features = nn.Dropout(cfg.dropout_features)

        self.logit_temp = cfg.logit_temp
        self.skip_masked = cfg.skip_masked
        self.skip_nomask = cfg.skip_nomask
        self.final_downsample = cfg.final_downsample

        if self.mask_before_cnn:
            self.mask_emb = nn.Parameter(torch.FloatTensor(cfg.feature_dim).uniform_())
        else:
            self.mask_emb = nn.Parameter(torch.FloatTensor(encoder_input_dim).uniform_())

        self.encoder = Zipformer2(
            output_downsampling_factor=2 if self.final_downsample else 1,
            downsampling_factor=_to_int_tuple(cfg.downsampling_factor),
            num_encoder_layers=_to_int_tuple(cfg.num_encoder_layers),
            encoder_dim=_to_int_tuple(cfg.encoder_dim),
            encoder_unmasked_dim=_to_int_tuple(cfg.encoder_unmasked_dim),
            query_head_dim=_to_int_tuple(cfg.query_head_dim),
            pos_head_dim=_to_int_tuple(cfg.pos_head_dim),
            value_head_dim=_to_int_tuple(cfg.value_head_dim),
            pos_dim=cfg.pos_dim,
            num_heads=_to_int_tuple(cfg.num_heads),
            feedforward_dim=_to_int_tuple(cfg.feedforward_dim),
            cnn_module_kernel=_to_int_tuple(cfg.cnn_module_kernel),
            dropout=ScheduledFloat((0.0, 0.3), (20000.0, 0.1)),
            warmup_batches=4000.0,
            causal=cfg.causal,
            chunk_size=_to_int_tuple(cfg.chunk_size),
            left_context_frames=_to_int_tuple(cfg.left_context_frames),
        )

        self.use_layer_norm = cfg.use_layer_norm
        if self.use_layer_norm:
            self.layer_norm = LayerNorm(self.embed)
        else:
            self.layer_norm = None

        self.untie_final_proj = cfg.untie_final_proj
        self.final_proj = nn.Linear(encoder_output_dim, sum(cfg.num_classes))

        # modules below are not needed during fine-tuning
        self.num_classes = cfg.num_classes
        self.pred_masked_weight = cfg.pred_masked_weight
        self.pred_nomask_weight = cfg.pred_nomask_weight
        self.loss_weights = cfg.loss_weights

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""

        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    def apply_mask(self, x, padding_mask, target_list, time_compress_rate=1):
        B, T, C = x.shape
        if self.mask_prob > 0:
            if time_compress_rate>1:
                assert isinstance(time_compress_rate, int), "time compress rate should be int"
                compress_T = math.ceil(T/time_compress_rate)
                sub_mask_indices = compute_mask_indices(
                    (B, compress_T),
                    padding_mask[:, ::time_compress_rate],
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.zeros(padding_mask.shape, dtype=torch.bool, device=x.device)
                sub_mask_indices = torch.from_numpy(sub_mask_indices).to(x.device)
                for repeat_step in range(time_compress_rate):
                    sub_compress_T = math.ceil((T-repeat_step)/time_compress_rate)
                    mask_indices[:, repeat_step::time_compress_rate] = sub_mask_indices[:, :sub_compress_T]
                x[mask_indices] = self.mask_emb.to(x.dtype)
            else:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
                x[mask_indices] = self.mask_emb.to(x.dtype)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices

    def forward_features(self, source: torch.Tensor, x_lens: torch.Tensor) -> torch.Tensor:
        features, x_lens = self.encoder_embed(source, x_lens)
        features = features.transpose(1, 2) # for consistence with original hubert cnn
        return features, x_lens

    def forward_targets(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        return features, target_list

    def forward_targets_and_mask(
        self,
        features: torch.Tensor,
        target_list: List[torch.Tensor],
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        # Trim features to ensure labels exist and then get aligned labels
        feat_tsz = features.size(2)
        targ_tsz = min([t.size(1) for t in target_list])
        # assert mask.shape[1] == targ_tsz
        if self.feat2tar_ratio * feat_tsz > targ_tsz:
            feat_tsz = int(targ_tsz / self.feat2tar_ratio)
            features = features[..., :feat_tsz]
        target_inds = torch.arange(feat_tsz).float() * self.feat2tar_ratio
        target_list = [t[:, target_inds.long()] for t in target_list]
        mask_inds = torch.arange(feat_tsz).float() * self.feature_ds_rate
        mask = mask[:, mask_inds.long()]
        
        return features, target_list, mask

    def forward_padding_mask(
        self,
        features: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        extra = padding_mask.size(1) % features.size(1)
        if extra > 0:
            padding_mask = padding_mask[:, :-extra]
        padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
        padding_mask = padding_mask.all(-1)
        return padding_mask

    def forward(
        self,
        source: torch.Tensor,
        target_list: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = True,
        features_only: bool = False,
    ):
        """output layer is 1-based"""
        if padding_mask is not None:
            x_lens = (~padding_mask).sum(dim=-1)
        else:
            x_lens = torch.ones((source.shape[0], ), device = source.device)*source.shape[1]
        if self.mask_before_cnn:
            if mask:
                source, mask_indices = self.apply_mask(source, padding_mask, target_list, self.feature_ds_rate)
            else:
                mask_indices = None

            features, _ = self.forward_features(source, x_lens)
            if target_list is not None:
                features, target_list, mask_indices = self.forward_targets_and_mask(features, target_list, mask_indices)

            features_pen = features.float().pow(2).mean()

            features = features.transpose(1, 2)
            if self.layer_norm is not None:
                features = self.layer_norm(features)
            unmasked_features = features.clone()

            if padding_mask is not None:
                padding_mask = self.forward_padding_mask(features, padding_mask)

            features = self.dropout_input(features)
            unmasked_features = self.dropout_features(unmasked_features)
            x = features
        else:
            features, _ = self.forward_features(source, x_lens)
            if target_list is not None:
                features, target_list = self.forward_targets(features, target_list)

            features_pen = features.float().pow(2).mean()

            features = features.transpose(1, 2)
            if self.layer_norm is not None:
                features = self.layer_norm(features)
            unmasked_features = features.clone()

            if padding_mask is not None:
                padding_mask = self.forward_padding_mask(features, padding_mask)

            features = self.dropout_input(features)
            unmasked_features = self.dropout_features(unmasked_features)

            if mask:
                x, mask_indices = self.apply_mask(features, padding_mask, target_list)
            else:
                x = features
                mask_indices = None

        # feature: (B, T, D), float
        # target: (B, T), long
        # x: (B, T, D), float -> (T, B, D), float
        # padding_mask: (B, T), bool
        # mask_indices: (B, T), bool

        x = x.transpose(0, 1)
        x, x_lens, layer_features = self.encoder(x, (~padding_mask).sum(dim=-1))
        x = x.transpose(0, 1)

        if features_only:
            return {"x": x, "x_lens": x_lens, "padding_mask": padding_mask, "features": features, "layer_features": layer_features}

        if not self.skip_masked:
            masked_indices = torch.logical_and(~padding_mask, mask_indices)
            proj_x_m = self.final_proj(x[masked_indices])
            proj_x_m /= self.logit_temp
            logit_m_list = [proj_x_m for _ in range(len(target_list))]
        else:
            logit_m_list = [None for _ in target_list]

        if not self.skip_nomask:
            nomask_indices = torch.logical_and(~padding_mask, ~mask_indices)
            proj_x_u = self.final_proj(x[nomask_indices])
            proj_x_u /= self.logit_temp
            logit_u_list = [proj_x_u for _ in range(len(target_list))]
        else:
            logit_u_list = [None for _ in target_list]

        # result = {
        #     "logit_m_list": logit_m_list,
        #     "logit_u_list": logit_u_list,
        #     "padding_mask": padding_mask,
        #     "features_pen": features_pen,
        # }
        targ_m_list = target_list[0][masked_indices]
        targ_m_list = targ_m_list.long()
        targ_m_list = [targ_m_list for _ in range(len(target_list))]

        targ_u_list = target_list[0][nomask_indices]
        targ_u_list = targ_u_list.long()
        targ_u_list = [targ_u_list for _ in range(len(target_list))]
        return self.compute_loss(
            logit_m_list, logit_u_list, targ_m_list, targ_u_list, features_pen
        )

    def extract_features(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        mask: bool = False,
        ret_conv: bool = False,
        output_layer: int = -1,
        do_final_down_sample = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        res = self.forward(
            source,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
        )
        length = res["x_lens"]
        if ret_conv:
            feature = res["features"]
        elif output_layer==-1 and do_final_down_sample:
            feature = res["x"]
        else:
            feature = res["layer_features"][output_layer]
            if output_layer>=0 and self.encoder.output_downsampling_factor == 2 and do_final_down_sample:
                feature = self.encoder.downsample_output(feature)
            length = (~res["padding_mask"]).sum(dim=-1)
            feature = feature.transpose(0, 1)

        return feature, res["padding_mask"], length

    def get_logits(self, net_output, is_masked=True):
        if is_masked:
            logits_list = net_output["logit_m_list"]
        else:
            logits_list = net_output["logit_u_list"]
        logits_list = [x.float() for x in logits_list if x is not None]
        return logits_list

    def get_targets(self, net_output, is_masked=True):
        logits_list = self.get_logits(net_output, is_masked)
        targets_list = [x.new_zeros(x.size(0), dtype=torch.long) for x in logits_list]
        return targets_list

    def get_extra_losses(self, net_output):
        extra_losses = []
        names = []

        if "features_pen" in net_output:
            extra_losses.append(net_output["features_pen"])
            names.append("features_pen")

        return extra_losses, names

    def remove_pretraining_modules(self):
        self.final_proj = None

    def compute_loss(
        self, logit_m_list, logit_u_list, targ_m_list, targ_u_list, features_pen
    ):
        loss = 0.0
        sample_size = 0
        logging_output = {}
        reduce = True
        reduction = "sum" if reduce else "none"

        loss_m_list = []
        logp_m_list = [x.float() for x in logit_m_list if x is not None]
        logp_m_list = torch.cat(logp_m_list)
        targ_m_list = torch.cat(targ_m_list)

        loss_m = F.cross_entropy(logp_m_list, targ_m_list, reduction=reduction)
        loss_m_list.append(loss_m)
        logging_output[f"loss_m_0"] = loss_m.detach().item()

        assert self.pred_masked_weight == 0 or len(logp_m_list) > 0
        if self.pred_masked_weight > 0:
            loss += self.pred_masked_weight * sum(loss_m_list)
            sample_size += len(targ_m_list)

        loss_u_list = []
        logp_u_list = [x.float() for x in logit_u_list if x is not None]
        logp_u_list = torch.cat(logp_u_list)
        targ_u_list = torch.cat(targ_u_list)

        loss_u = F.cross_entropy(logp_u_list, targ_u_list, reduction=reduction)
        loss_u_list.append(loss_u)
        logging_output[f"loss_u_0"] = loss_u.detach().item()

        assert self.pred_nomask_weight == 0 or len(logp_u_list) > 0
        if self.pred_nomask_weight > 0:
            loss += self.pred_nomask_weight * sum(loss_u_list)
            sample_size += len(targ_u_list)

        if self.loss_weights is not None:
            extra_losses = []
            names = []
            extra_losses.append(features_pen)
            names.append("features_pen")
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
                names = [names]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, n, coef in zip(extra_losses, names, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    logging_output[f"loss_{n}"] = p.item()

        logging_output = {
            "loss": loss.item() if reduce else loss,
            **logging_output,
        }

        # for lk in self.log_keys:
        #     if lk in net_output:
        #         logging_output[lk] = float((net_output[lk]))

        def compute_correct(logits, target):
            if logits.numel() == 0:
                return 0, 0
            else:
                assert logits.dim() > 1, logits.shape
                max = logits.argmax(-1) == target
                min = logits.argmin(-1) == target
                both = max & min
                corr = max.long().sum().item() - both.long().sum().item()
                count = max.numel()
                return corr, count

        with torch.no_grad():
            corr_m, count_m = compute_correct(logp_m_list, targ_m_list)
            logging_output[f"correct_m_0"] = corr_m
            logging_output[f"count_m_0"] = count_m

            corr_u, count_u = compute_correct(logp_u_list, targ_u_list)
            logging_output[f"correct_u_0"] = corr_u
            logging_output[f"count_u_0"] = count_u

        return loss, sample_size, logging_output
