# Copyright 2023 Garena Online Private Limited
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

"""Loss functions used in the paper "Fast Diffusion Model"."""

import torch
from torch_utils import persistence
from torch_utils import distributed as dist
import numpy as np

#----------------------------------------------------------------------------
# VP loss function with FDM loss weight warmup strategy.

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5, warmup_ite=None):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t
        self.warmup_ite = warmup_ite
        self.clamp_cur = 5.
        self.clamp_max = 500.
        if self.warmup_ite:
            self.warmup_step = np.exp(np.log(100) / self.warmup_ite)

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        if self.warmup_ite:
            if self.clamp_cur < self.clamp_max:
                weight.clamp_max_(self.clamp_cur)
                self.clamp_cur *= self.warmup_step
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# VE loss function with FDM loss weight warmup strategy.

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100, warmup_ite=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.warmup_ite = warmup_ite
        self.clamp_cur = 5.
        self.clamp_max = 500.
        if self.warmup_ite:
            self.warmup_step = np.exp(np.log(100) / self.warmup_ite)

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        if self.warmup_ite:
            if self.clamp_cur < self.clamp_max:
                weight.clamp_max_(self.clamp_cur)
                self.clamp_cur *= self.warmup_step
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# EDM loss function with our FDM weight warmup strategy.

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, warmup_ite=None):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.warmup_ite = warmup_ite
        self.clamp_cur = 5.
        self.clamp_max = 500.
        if self.warmup_ite:
            self.warmup_step = np.exp(np.log(100) / self.warmup_ite)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        if self.warmup_ite:
            if self.clamp_cur < self.clamp_max:
                weight.clamp_max_(self.clamp_cur)
                self.clamp_cur *= self.warmup_step
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss
