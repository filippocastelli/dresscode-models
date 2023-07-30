from enum import Enum

import torch
import torch.nn as nn

class ResBlockType(Enum):
    DOWN = 1
    UP = 2
    SAME = 3

class ResBlock(nn.Module):
    """ Residual Block """
    """ https://arxiv.org/abs/1512.03385 """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            type: ResBlockType = ResBlockType.DOWN,
            norm_layer: nn.Module = nn.BatchNorm2d,
            ):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.type = type
        self.norm_layer = norm_layer
        
        super(ResBlock, self).__init__()
        self.using_bias = norm_layer == nn.InstanceNorm2d

        self.residual_link = self.get_residual_link()
        self.convolutional_block = self.get_convolutional_block()
        self.relu = nn.ReLU(inplace=True)
    

    def get_residual_link(self) -> nn.Module:
        if self.type == ResBlockType.SAME:
            return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
        elif self.type == ResBlockType.UP:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=True)
            )
        elif self.type == ResBlockType.DOWN:
            return nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=2, padding=1, bias=self.using_bias)
    
    def get_convolutional_block(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=self.using_bias),
            self.norm_layer(self.out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1, bias=self.using_bias),
            self.norm_layer(self.out_channels)
        )
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_link(x)
        return self.relu(residual + self.convolutional_block(residual))