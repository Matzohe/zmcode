# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# github@OpenGVLab,repo:LLaMA-Adapter, https://github.com/OpenGVLab/LLaMA-Adapter.git

import math

def adjust_learning_rate(optimizer, step, warmup_steps, lr, min_lr, steps):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if step < warmup_steps:
        lr = lr * step / warmup_steps + min_lr * (1 - step / warmup_steps)
    else:
        lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (step - warmup_steps) / (steps - warmup_steps)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr
