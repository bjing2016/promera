# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from scipy.stats import truncnorm
import torch

# Standard deviation of a standard normal truncated to [-2, 2]. Equal to
# scipy.stats.truncnorm.std(a=-2, b=2); computed once at import so the per-tensor
# init below does not call into scipy on the hot path.
_TRUNCNORM_STD = float(truncnorm.std(a=-2, b=2, loc=0, scale=1))


def _calculate_fan(linear_weight_shape, fan="fan_in"):
    fan_out, fan_in = linear_weight_shape

    if fan == "fan_in":
        f = fan_in
    elif fan == "fan_out":
        f = fan_out
    elif fan == "fan_avg":
        f = (fan_in + fan_out) / 2
    else:
        raise ValueError("Invalid fan option")

    return f


def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):
    # Truncated-normal init matching the original scipy implementation, but
    # vectorized in torch. scipy.truncnorm.rvs generates samples on the CPU via
    # numpy (and then copies them into the tensor) regardless of device, so for a
    # ~470M-param model it cost ~35s of pure CPU sampling at construction time.
    # torch.nn.init.trunc_normal_ is a single fused kernel that runs natively on
    # whatever device the tensor lives on.
    f = _calculate_fan(weights.shape, fan)
    scale = scale / max(1, f)
    std = math.sqrt(scale) / _TRUNCNORM_STD
    with torch.no_grad():
        torch.nn.init.trunc_normal_(
            weights, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std
        )


def lecun_normal_init_(weights):
    trunc_normal_init_(weights, scale=1.0)


def he_normal_init_(weights):
    trunc_normal_init_(weights, scale=2.0)


def glorot_uniform_init_(weights):
    torch.nn.init.xavier_uniform_(weights, gain=1)


def final_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def gating_init_(weights):
    with torch.no_grad():
        weights.fill_(0.0)


def bias_init_zero_(bias):
    with torch.no_grad():
        bias.fill_(0.0)


def bias_init_one_(bias):
    with torch.no_grad():
        bias.fill_(1.0)


def normal_init_(weights):
    torch.nn.init.kaiming_normal_(weights, nonlinearity="linear")
