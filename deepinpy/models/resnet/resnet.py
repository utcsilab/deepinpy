#!/usr/bin/env python

import torch.nn

from deepinpy.utils import utils


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

        ndims = len(x.shape)
        permute_shape = list(range(ndims))
        permute_shape.insert(1, permute_shape.pop(-1))
        x = x.permute(permute_shape)
        temp_shape = x.shape
        x = x.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))


        x = self.model(x)

        x = x.reshape(temp_shape) # 1, 2, 3, 12, 13
        permute_shape = list(range(ndims)) # we want 0,2,3,4,1
        permute_shape.insert(ndims-1, permute_shape.pop(1))
        x = x.permute(permute_shape)

        return x


class ResNetBlock(torch.nn.Module):
    def __init__(self, in_channels=2, latent_channels=64, out_channels=64, kernel_size=3, bias=False, batch_norm=True, final_relu=True, dropout=0):
        super(ResNetBlock, self).__init__()

        self.batch_norm = batch_norm
        self.final_relu = final_relu

        # initialize conv variables
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.bias = bias
        if dropout > 0:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.in_channels == self.out_channels:
            self.conv0 = None
        else:
            self.conv0 = self._conv_zero(self.in_channels, self.out_channels)
        self.conv1 = self._conv(self.in_channels, self.latent_channels)
        self.conv2 = self._conv(self.latent_channels, self.out_channels)

        if self.batch_norm:
            self.bn1 = self._bn(self.in_channels)
            self.bn2 = self._bn(self.latent_channels)

        self.relu = self._relu()

    def forward(self, x):
        if self.conv0:
            residual = self.conv0(x)
        else:
            residual = x

        out = x

        if self.batch_norm:
            out = self.bn1(out)

        out = self.relu(out)

        out = self.conv1(out)

        if self.dropout is not None:
            out = self.dropout(out)

        if self.batch_norm:
            out = self.bn2(out)

        if self.final_relu:
            out = self.relu(out)

        out = self.conv2(out)

        if self.dropout is not None:
            out = self.dropout(out)

        out += residual

        return out

    def _conv(self, in_channels, out_channels):
        return Conv2dSame(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                bias=self.bias)

    def _conv_zero(self, in_channels, out_channels):
        return Conv2dSame(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=self.bias)

    def _bn(self, channels):
        return torch.nn.BatchNorm2d(channels)

    def _relu(self):
        #return torch.nn.ReLU(inplace=True)
        return torch.nn.ReLU()

class ResNet(torch.nn.Module):
    def __init__(self, in_channels=2, latent_channels=64, num_blocks=3, kernel_size=7, bias=False, batch_norm=True, dropout=0, topk=None, l1lam=None):
        super(ResNet, self).__init__()

        self.batch_norm = batch_norm
        self.num_blocks = num_blocks

        # initialize conv variables
        self.in_channels = in_channels
        
        self.latent_channels = latent_channels
        self.kernel_size = kernel_size
        self.bias = bias
        self.dropout = dropout

        self.ResNetBlocks = self._build_model()
        #self.ResNetBlocks = self._build_model_bottleneck()

        #self.weights = self.ResNetBlocks[self.num_blocks // 2].conv2.weight

        self.l1lam = l1lam
        if self.l1lam:
            #self.threshold = torch.nn.Threshold(self.l1lam, 0)
            self.threshold = torch.nn.Softshrink(self.l1lam)

        self.topk = topk

    def forward(self, x):
        x = torch.view_as_real(x)
        ndims = len(x.shape)
        permute_shape = list(range(ndims))
        permute_shape.insert(1, permute_shape.pop(-1))
        x = x.permute(permute_shape)
        temp_shape = x.shape
        x = x.reshape((x.shape[0], -1, x.shape[-2], x.shape[-1]))
        
        #x = x.permute(0, 3, 1, 2)
        residual = x
        for n in range(self.num_blocks):
            x = self.ResNetBlocks[n](x)
            if n == self.num_blocks // 2:
                act = x
                if self.l1lam:
                    act = self.threshold(act)
                if self.topk:
                    act = utils.topk(act, self.topk, dim=1)
                x = act
        x += residual
        #return x.permute(0, 2, 3, 1), act
              
        x = x.reshape(temp_shape) # 1, 2, 3, 12, 13
        permute_shape = list(range(ndims)) # we want 0,2,3,4,1
        permute_shape.insert(ndims-1, permute_shape.pop(1))
        x = x.permute(permute_shape)
        return torch.view_as_complex(x.contiguous())

    def _build_model(self):
        ResNetBlocks = torch.nn.ModuleList()
        
        # first block goes from input space (2ch) to latent space (64ch)
        ResNetBlocks.append(self._add_block(final_relu=True, in_channels=self.in_channels, latent_channels=self.latent_channels, out_channels=self.latent_channels))

        # middle blocks go from latent space to latent space 
        for n in range(self.num_blocks - 2):
            ResNetBlocks.append(self._add_block(final_relu=True, in_channels=self.latent_channels, latent_channels=self.latent_channels, out_channels=self.latent_channels))

        # last block goes from latent space to output space (2ch) with no ReLU
        ResNetBlocks.append(self._add_block(final_relu=False, in_channels=self.latent_channels, latent_channels=self.latent_channels, out_channels=self.in_channels))

        return ResNetBlocks

    def _add_block(self, in_channels, latent_channels, out_channels, final_relu=True):
        return ResNetBlock(in_channels=in_channels,
                latent_channels=latent_channels,
                out_channels=out_channels,
                kernel_size=self.kernel_size,
                bias=self.bias,
                batch_norm=self.batch_norm,
                final_relu=final_relu, dropout=self.dropout)



