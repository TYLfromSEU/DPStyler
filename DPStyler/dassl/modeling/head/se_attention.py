import functools
import torch.nn as nn
from math import sqrt
import torch
import einops
from .build import HEAD_REGISTRY

class SE_attn(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):

        super(SE_attn, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.out_features=num_channels

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels)
        :return: output tensor
        """
        batch_size, num_channels= input_tensor.size()
        # # Average along each channel
        # squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation

        fc_out_1 = self.relu(self.fc1(input_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))
        #权重乘原向量
        weighted_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels))
        output_tensor=input_tensor+weighted_tensor

        return output_tensor  #

@HEAD_REGISTRY.register()
def se_attn_sr(num_channels, reduction_ratio=2,**kwargs):
    return SE_attn(num_channels, reduction_ratio,**kwargs)