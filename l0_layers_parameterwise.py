import math

import torch
from torch import nn
from torch.nn import functional as F

EPS = 1e-6

def hard_sigmoid(x):
    return torch.min(torch.max(x, torch.zeros_like(x)), torch.ones_like(x))


class _L0Norm(nn.Module):
    def __init__(self, origin, loc_mean=0, loc_sdev=0.01, beta=2 / 3, gamma=-0.1,
                 zeta=1.1, fix_temp=True):
        """
        Base class of layers using L0 Norm
        :param origin: original layer such as nn.Linear(..), nn.Conv2d(..)
        :param loc_mean: mean of the normal distribution which generates initial location parameters
        :param loc_sdev: standard deviation of the normal distribution which generates initial location parameters
        :param beta: initial temperature parameter
        :param gamma: lower bound of "stretched" s
        :param zeta: upper bound of "stretched" s
        :param fix_temp: True if temperature is fixed
        """
        super(_L0Norm, self).__init__()
        self._origin = origin
        self._size = self._origin.weight.size()
        self.loc = nn.Parameter(torch.zeros(self._size).normal_(loc_mean, loc_sdev))
        self.temp = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
        self.register_buffer("uniform", torch.zeros(self._size))
        if self._origin.bias is not None:
            self._size_bias = self._origin.bias.size()
            self.loc_bias = nn.Parameter(torch.zeros(self._size_bias).normal_(loc_mean, loc_sdev))
            self.temp_bias = beta if fix_temp else nn.Parameter(torch.zeros(1).fill_(beta))
            self.register_buffer("uniform_bias", torch.zeros(self._size_bias))

        self.gamma = gamma
        self.zeta = zeta
        self.gamma_zeta_ratio = math.log(-gamma / zeta)

    def _get_mask(self):
        s_bias = None
        if self.training:
            self.uniform.uniform_(EPS, 1-EPS)
            u = self.uniform
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.loc) / self.temp)
            s = s * (self.zeta - self.gamma) + self.gamma
            penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
            if self._origin.bias is not None:
                self.uniform_bias.uniform_(EPS, 1-EPS)
                u_bias = self.uniform_bias
                s_bias = torch.sigmoid((torch.log(u_bias) - torch.log(1 - u_bias) +
                    self.loc_bias) / self.temp_bias)
                s_bias = s_bias * (self.zeta - self.gamma) + self.gamma
                penalty += torch.sigmoid(self.loc_bias - self.temp_bias * self.gamma_zeta_ratio).sum()
        else:
            s = torch.sigmoid(self.loc) * (self.zeta - self.gamma) + self.gamma
            if self._origin.bias is not None:
                s_bias = torch.sigmoid(self.loc_bias) * (self.zeta - self.gamma) + self.gamma
            penalty = 0
        weight_mask = hard_sigmoid(s)
        bias_mask = hard_sigmoid(s_bias) if s_bias is not None else None
        return weight_mask, bias_mask#, penalty

    def _get_panalty(self):
        penalty = torch.sigmoid(self.loc - self.temp * self.gamma_zeta_ratio).sum()
        if self._origin.bias is not None:
            penalty += torch.sigmoid(self.loc_bias - self.temp_bias *
                self.gamma_zeta_ratio).sum()
        return self.lamba * penalty

    def regularization(self):
        return self._get_panalty()

    def constrain_parameters(self, **kwargs):
        self.loc.data.clamp_(min=math.log(1e-2), max=math.log(1e2))
        if self._origin.bias is not None:
            self.loc_bias.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

class L0DenseParam(_L0Norm):
    def __init__(self, in_features, out_features, bias=True, bias_l0=True, #weight_decay=1.,
             droprate_init=0.5, temperature=2./3.,
                              lamba=1., #local_rep=False,
                              **kwargs):
        super(L0DenseParam, self).__init__(
                nn.Linear(in_features, out_features, bias=(bias and bias_l0)),
                loc_mean=droprate_init, beta=temperature, **kwargs)
        self.random = nn.Linear(in_features, out_features, bias=bias)
        self.random.weight.requires_grad = False
        if bias:
            self.random.bias.requires_grad = False
        self.lamba = lamba

    def forward(self, input):
        weight_mask, bias_mask = self._get_mask()
        origin_bias = self._origin.bias * bias_mask if self._origin.bias is not None else None
        origin_output = F.linear(input, self._origin.weight * weight_mask, origin_bias)
        random_output = self.random(input)
        if self.training:
            return origin_output + random_output#, penalty # TODO need lambda
        else:
            return origin_output


class L0Conv2dParam(_L0Norm):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
        dilation=1, groups=1, bias=True, bias_l0=True,
                         droprate_init=0.5, temperature=2./3.,# weight_decay=1.,
                         lamba=1., #local_rep=False,
                         **kwargs):

        super(L0Conv2dParam, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            dilation=dilation,groups=groups, bias=bias and bias_l0),
                loc_mean=droprate_init, beta=temperature, **kwargs)

        self.random = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                            dilation=dilation,groups=groups, bias=bias)
        self.random.weight.requires_grad = False
        if bias:
            self.random.bias.requires_grad = False
        self.lamba = lamba

    def forward(self, input):
        weight_mask, bias_mask = self._get_mask()
        origin_bias = self._origin.bias * bias_mask if self._origin.bias is not None else None
        origin_conv = F.conv2d(input, self._origin.weight * weight_mask,origin_bias, stride=self._origin.stride,
                        padding=self._origin.padding, dilation=self._origin.dilation, groups=self._origin.groups)
        random_conv = self.random(input)
        if self.training:
            return origin_conv + random_conv#, penalty
        else:
            return origin_conv
