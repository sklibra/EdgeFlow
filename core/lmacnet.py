import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

def channel_rearrange(x: Tensor) -> Tensor:

    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // 2

    # reshape
    # [batch_size, num_channels, height, width] -> [batch_size, groups, channels_per_group, height, width]
    x = x.view(batch_size, channels_per_group, 2, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batch_size, -1, height, width)

    return x


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()

        branch_dim = dim // 2
        self.conv1 = nn.Conv2d(branch_dim, branch_dim, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(branch_dim, branch_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(branch_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:

        x = channel_rearrange(x)
        x1, x2 = x.chunk(2, dim=1)
        x2 = self.norm(self.conv3(x2))
        x2 = self.act(self.conv1(x2))
        out = torch.cat((x1, x2), dim=1)
  
        return out


class LMAC(nn.Module):
    def __init__(self, in_c: int = 3, out_c: int = 1000, depths: list = None,
                 dims: list = None):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_c, dims[0], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True)
        )

        self.trunk_layers = nn.ModuleList()  # 3 intermediate downsampling conv layers
 
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.Conv2d(dims[i], dims[i], kernel_size=3, stride=2, padding=1, groups=dims[i], bias=False),
                nn.BatchNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i], kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True)
                )
            self.trunk_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple blocks

        for i in range(3):
            stage = nn.Sequential(
                *[Block(dim=dims[i])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)

        self.head = nn.Conv2d(dims[-1], out_c, kernel_size=1, stride=1, padding=0)

        
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for i in range(3):
            x = self.trunk_layers[i](x)
            x1 = self.stages[i](x)
            x = torch.cat((x, x1), dim=1)
            x = channel_rearrange(x)

        return x  

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)
        
        x = self.forward_features(x)

        x = self.head(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x

     


def LMAC_tiny(out_c: int):
    model = LMAC(depths=[1, 3, 1],
                     dims=[16, 32, 64, 128],
                     out_c=out_c
                     )
    return model

def LMAC_small(out_c: int, dropout: float):
    model = LMAC(depths=[1, 3, 1],
                     dims=[32, 64, 128, 256],
                     out_c=out_c, 
                     dropout=dropout)
    return model


def LMAC_base(out_c: int, dropout: float):
    model = LMAC(depths=[1, 3, 1],
                     dims=[64, 128, 256, 512],
                     out_c=out_c, 
                     dropout=dropout)
    return model

