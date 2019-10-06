#!/usr/bin/env python

import numpy as np
import torch
#import torch.nn.functional
import cfl
import sys

import deepinpy.utils.complex as cp


class Conv2dSame(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True, padding_layer=torch.nn.ReflectionPad2d):
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka,kb,ka,kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )
    def forward(self, x):
        return self.net(x)


class ResNet5Block(torch.nn.Module):
    def __init__(self, num_filters=32, filter_size=3, T=4, num_filters_start=2, num_filters_end=2, batch_norm=False):
        super(ResNet5Block, self).__init__()
        num_filters_start = num_filters_end = 2
        if batch_norm:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.BatchNorm2d(num_filters),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters_end,filter_size)
            )
        else:
            self.model = torch.nn.Sequential(
                Conv2dSame(num_filters_start,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters,filter_size),
                torch.nn.ReLU(),
                Conv2dSame(num_filters,num_filters_end,filter_size)
            )
        self.T = T
        
    def forward(self,x,device='cpu'):
        return x + self.step(x, device=device)
    
    def step(self, x, device='cpu'):
        # reshape (batch,x,y,channel=2) -> (batch,channe=2,x,y)
        x = x.permute(0, 3, 1, 2)
        y = self.model(x)
        return y.permute(0, 2, 3, 1)
