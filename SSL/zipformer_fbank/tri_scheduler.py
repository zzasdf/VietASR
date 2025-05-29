# Copyright    2025  Shanghai Jiao Tong University  (authors: Jianheng Zhuo)
#
# See ../LICENSE for clarification regarding multiple authors
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

import math
from optim import LRScheduler


class TriStageLRSchedule(LRScheduler):
    """Tristage learning rate schedulr

    Implement the learning rate scheduler in https://arxiv.org/pdf/1904.08779.pdf

    Similar to inverse_squre_root scheduler, but tri_stage learning rate employs
    three stages LR scheduling:

        - warmup stage, starting from `lr` * `init_lr_scale`, linearly
          increased to `lr` in `warmup_steps` iterations

        - hold stage, after `warmup_steps`, keep the LR as `lr` for `hold_steps`
          iterations

        - decay stage, after hold stage, decay LR exponetially to
          `lr` * `final_lr_scale` in `decay_steps`;
          after that LR is keep as `final_lr_scale` * `lr`

    During warmup::

      init_lr = cfg.init_lr_scale * cfg.lr
      lrs = torch.linspace(init_lr, cfg.lr, cfg.warmup_steps)
      lr = lrs[update_num]

    During hold::

      lr = cfg.lr

    During decay::

      decay_factor = - math.log(cfg.final_lr_scale) / cfg.decay_steps
      lr = cfg.lr * exp(- (update_num - warmup_steps - decay_steps) * decay_factor)

    After that::

      lr = cfg.lr * cfg.final_lr_scale
    """

    def __init__(
        self,
        optimizer,
        init_lr_scale=0.01,
        final_lr_scale=0.01,
        verbose: bool = False,
        max_update: int = 0,
        warmup_steps: int = 0,
        hold_steps: int = 0,
        decay_steps: int = 0,
        phase_ratio=None,
    ):
        super(TriStageLRSchedule, self).__init__(optimizer, verbose)
        # super().__init__(cfg, optimizer)
        # calculate LR at each point
        peak_lr = self.base_lrs[0]
        self.peak_lr = peak_lr
        self.init_lr = init_lr_scale * peak_lr
        self.final_lr = final_lr_scale * peak_lr

        if phase_ratio is not None:
            assert max_update > 0
            assert sum(phase_ratio) == 1, "phase ratios must add up to 1"
            self.warmup_steps = int(max_update * phase_ratio[0])
            self.hold_steps = int(max_update * phase_ratio[1])
            self.decay_steps = int(max_update * phase_ratio[2])
        else:
            self.warmup_steps = warmup_steps
            self.hold_steps = hold_steps
            self.decay_steps = decay_steps

        assert (
            self.warmup_steps + self.hold_steps + self.decay_steps > 0
        ), "please specify steps or phase_ratio"

        self.warmup_rate = (
            (self.peak_lr - self.init_lr) / self.warmup_steps
            if self.warmup_steps != 0
            else 0
        )
        self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        # initial learning rate
        # self.lr = self.init_lr
        # self.optimizer.set_lr(self.lr)

    def _decide_stage(self, update_step):
        """
        return stage, and the corresponding steps within the current stage
        """
        if update_step < self.warmup_steps:
            # warmup state
            return 0, update_step

        offset = self.warmup_steps

        if update_step < offset + self.hold_steps:
            # hold stage
            return 1, update_step - offset

        offset += self.hold_steps

        if update_step <= offset + self.decay_steps:
            # decay stage
            return 2, update_step - offset

        offset += self.decay_steps

        # still here ? constant lr stage
        return 3, update_step - offset


    def get_lr(self):
        """Update the learning rate after each update."""
        stage, steps_in_stage = self._decide_stage(self.batch)
        lr = []
        peak_lr = self.base_lrs[0]
        if stage == 0:
            lr.append(self.init_lr + self.warmup_rate * steps_in_stage)
        elif stage == 1:
            lr.append(peak_lr)
        elif stage == 2:
            lr.append(peak_lr * math.exp(-self.decay_factor * steps_in_stage))
        elif stage == 3:
            lr.append(self.final_lr)
        else:
            raise ValueError("Undefined stage")

        # self.optimizer.set_lr(self.lr)

        return lr
        # return self.lr


