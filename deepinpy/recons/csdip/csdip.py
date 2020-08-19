import sys

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import DCGAN_MRI, decodernw
from deepinpy.recons import Recon
import numpy as np
import torch


class CSDIPRecon(Recon):

    def __init__(self, args):
        super(CSDIPRecon, self).__init__(args)

        self.N1 = 8
        self.N2 = 8
        self.x_adj = None

        self.output_size = self.D.shape[1:]
        print('output size:', self.output_size)

        if len(self.output_size) > 2:
            self.num_output_channels = 2 * np.prod(self.output_size[:-2])
            self.output_size = self.output_size[-2:]
        else:
            self.num_output_channels = 2

        if self.hparams.network == 'DCGAN':
            # FIXME: make work for arbitrary input sizes
            self.network = DCGAN_MRI(self.hparams.z_dim, ngf=64, output_size=self.output_size, nc=2, num_measurements=256)

        elif self.hparams.network == 'DeepDecoder':

            # initial number of channels given by z_dim
            self.num_channels_up = [self.hparams.z_dim] + [self.hparams.latent_channels]*(self.hparams.num_blocks - 1)

            # FIXME: make generic for number of dimensions
            scale_x = [int(np.product([self.N1] + [np.exp(np.log(self.output_size[0]/self.N1)/self.hparams.num_blocks)] * i)) for i in range(self.hparams.num_blocks)] + [self.output_size[0]]
            scale_y = [int(np.product([self.N2] + [np.exp(np.log(self.output_size[1]/self.N2)/self.hparams.num_blocks)] * i)) for i in range(self.hparams.num_blocks)] + [self.output_size[1]]

            self.upsample_size = list(zip(scale_x, scale_y))

            self.network = decodernw(num_output_channels=self.num_output_channels, num_channels_up=self.num_channels_up, upsample_first=True, need_sigmoid=False, upsample_size=self.upsample_size)
        else:
            # FIXME: error logging
            print('ERROR: invalid network specified')
            sys.exit(-1)

        self.zseed = None
        self.use_cpu = args.cpu

    def batch(self, data):
        maps = data['maps']
        masks = data['masks']
        inp = data['out'] #read in maps, masks, and k-space input

        # initialize z vector only once per index
        # FIXME: only works for num_data_sets=1
        if self.zseed is None:
            if self.hparams.network == 'DCGAN':
                zseed = torch.zeros(self.batch_size*self.hparams.z_dim).view(self.batch_size,self.hparams.z_dim,1,1)
            else:
                zseed = torch.zeros(self.batch_size, self.hparams.z_dim, self.N1, self.N2)
                print('zseed shape is:', zseed.shape)
            zseed.data.normal_().type(torch.FloatTensor)
            if not self.use_cpu:
                zseed = zseed.to(inp.device)
            self.zseed = zseed

        self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.use_sigpy, noncart=self.noncart)
    def forward(self, y):
        out =  self.network(self.zseed) #DCGAN acts on the low-dim space parameterized by z to output the image x
        if self.hparams.network == 'DeepDecoder':
            if len(self.D.shape) == 4:
                out = out.reshape(out.shape[0], 2, -1, out.shape[-2], out.shape[-1])
                out = out.permute((0, 2, 3, 4, 1))
            else:
                out = out.permute(0, 2, 3, 1)
        return out

    def get_metadata(self):
        return {}
