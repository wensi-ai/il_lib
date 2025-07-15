"""
Weight standardization conv layers: http://arxiv.org/abs/1903.10520
"""

import torch.nn as nn
import torch.nn.functional as F


class Conv1dWS(nn.Conv1d):
    """
    Weight Standardization
    """

    def forward(self, x):
        weight = self.weight
        N = weight.size(0)  # batch dim
        weight_mean = weight.view(N, -1).mean(dim=1)
        weight = weight - weight_mean.view(N, 1, 1)
        std = weight.view(N, -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Conv2dWS(nn.Conv2d):
    """
    Weight Standardization
    """

    def forward(self, x):
        weight = self.weight
        N = weight.size(0)  # batch dim
        weight_mean = weight.view(N, -1).mean(dim=1)
        weight = weight - weight_mean.view(N, 1, 1, 1)
        std = weight.view(N, -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class Conv3dWS(nn.Conv3d):
    """
    Weight Standardization
    """

    def forward(self, x):
        weight = self.weight
        N = weight.size(0)  # batch dim
        weight_mean = weight.view(N, -1).mean(dim=1)
        weight = weight - weight_mean.view(N, 1, 1, 1, 1)
        std = weight.view(N, -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )