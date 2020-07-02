import sys

from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import DCGAN_MRI, decodernw
from deepinpy.recons import Recon
import torch


class CSDIPRecon(Recon):

    def __init__(self, args):
        super(CSDIPRecon, self).__init__(args)
        self.Z_DIM = self.hparams.latent_channels
        self.x_adj = None

        # FIXME: make work for arbitrary input sizes
        if self.hparams.network == 'DCGAN':
            self.network = DCGAN_MRI(self.Z_DIM, ngf=64, output_size=[320, 256], nc=2, num_measurements=256)
        elif self.hparams.network == 'DeepDecoder':
            self.num_channels_up = [self.Z_DIM]*6
            self.network = decodernw(num_output_channels=2, num_channels_up=self.num_channels_up, upsample_first=True, need_sigmoid=False)
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
                total_upsample = 2**(len(self.num_channels_up))
                if total_upsample > 64:
                    raise ValueError('desired output size of [320,256] is incompatible with more than 64x upsampling')
                zseed = torch.zeros(self.batch_size,self.num_channels_up[0],320//total_upsample,256//total_upsample)
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

