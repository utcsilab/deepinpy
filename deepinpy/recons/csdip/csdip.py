import sys

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import DCGAN_MRI, decodernw
from deepinpy.recons import Recon
import numpy as np
import torch


class CSDIPRecon(Recon):

    def __init__(self, args):
        super(CSDIPRecon, self).__init__(args)
        self.Z_DIM = 16
        self.N1 = 4
        self.N2 = 4
        self.x_adj = None


        self.output_size = self.D.shape[1:]
        print('output size:', self.output_size)

        # FIXME: make work for arbitrary input sizes
        if self.hparams.network == 'DCGAN':
            self.network = DCGAN_MRI(self.Z_DIM, ngf=64, output_size=self.output_size, nc=2, num_measurements=256)
        elif self.hparams.network == 'DeepDecoder':
            num_blocks = self.hparams.num_blocks


            self.num_channels_up = [self.Z_DIM] + [self.hparams.latent_channels]*(self.hparams.num_blocks - 1)
            #print(self.num_channels_up)

            scale_x = [np.round(np.product([self.N1] + [np.exp(np.log(self.output_size[0]/self.N1)/self.hparams.num_blocks)] * i)) for i in range(self.hparams.num_blocks + 1)]
            scale_y = [np.round(np.product([self.N2] + [np.exp(np.log(self.output_size[1]/self.N2)/self.hparams.num_blocks)] * i)) for i in range(self.hparams.num_blocks + 1)]

            self.upsample_size = list(zip([int(s_x) for s_x in scale_x], [int(s_y) for s_y in scale_y]))

            self.network = decodernw(num_output_channels=2, num_channels_up=self.num_channels_up, upsample_first=True, need_sigmoid=False, upsample_size=self.upsample_size)
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
                zseed = torch.zeros(self.batch_size*self.Z_DIM).view(self.batch_size,self.Z_DIM,1,1)
            else:
                zseed = torch.zeros(self.batch_size, self.Z_DIM, self.N1, self.N2)
                print('zseed shape is:', zseed.shape)
            if self.use_cpu:
                zseed.data.normal_().type(torch.FloatTensor)
            else:
                zseed.data.normal_().type(torch.cuda.FloatTensor)
                zseed = zseed.to(inp.device)
            self.zseed = zseed

        self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.use_sigpy, noncart=self.noncart)

    def forward(self, y):
        out =  self.network(self.zseed) #DCGAN acts on the low-dim space parameterized by z to output the image x
        if self.hparams.network == 'DeepDecoder':
            out = out.permute(0, 2, 3, 1)
        return out

    def get_metadata(self):
        return {}


#Gotta be careful with the set of arguments for the DCGAN. In the original code, we have the following:
#args.Z_DIM, NGF, args.IMG_SIZE,\
#            args.NUM_CHANNELS, args.NUM_MEASUREMENTS

#data['imgs'].shape

