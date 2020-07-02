import torch
import torch.nn as nn
import torch.nn.functional as F

from deepinpy.utils import utils


class DCGAN_MRI(nn.Module):
    def __init__(self, nz, ngf=64, output_size=[320,256], nc=2, num_measurements=1000):
        super(DCGAN_MRI, self).__init__()
        self.nc = nc
        self.output_size = output_size

        ratio = output_size[0] / output_size[1]
        if ratio < 1:
            d2 = 5
            d1 = int(d2 * ratio)
        else:
            d1 = 5
            d2 = int(d1 / ratio)

        #print('ratio, d1, d2', d1, d2)
        #print('output_shape:', output_size)

        self.conv1 = nn.ConvTranspose2d(nz, ngf, [d1, d2], 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf)
        self.conv2 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.conv3 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)
        self.conv4 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        self.conv5 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        self.conv6 = nn.ConvTranspose2d(ngf, ngf, 6, 2, 2, bias=False)
        self.bn6 = nn.BatchNorm2d(ngf)
        self.conv7 = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False) #output is image
    
    def forward(self, z):
        input_size = z.size()
        #print(input_size)
        x = F.relu(self.bn1(self.conv1(z)))
        #print(x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print(x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print(x.shape)
        x = F.relu(self.bn4(self.conv4(x)))
        #print(x.shape)
        x = F.relu(self.bn5(self.conv5(x)))
        #print(x.shape)
        x = F.relu(self.bn6(self.conv6(x)))
        #print(x.shape)
        x = self.conv7(x)
        #print(x.shape)
        #x = torch.tanh(x)
        #print(x.shape)
        x = x.permute(0, 2, 3, 1)
        return x

    #notice in the resnet model that there is a function called step(self,x,device='cpu')
