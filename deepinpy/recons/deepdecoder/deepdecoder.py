from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import decodernw
from deepinpy.recons import Recon
import torch

class DeepDecoderRecon(Recon):

    def __init__(self, args):
        self.num_channels_up = [128]*6
        self.x_adj = None
        super(DeepDecoderRecon, self).__init__(args)

        self.denoiser = decodernw(num_output_channels=2,num_channels_up=self.num_channels_up, upsample_first=True, need_sigmoid=False) #todo: check whether I need to use sigmoid


        # warning: don't use self.cpu, it's a function in deepdecoder
        self.use_cpu = args.cpu
        if self.use_cpu:
            pass
        else:
            self.denoiser = self.denoiser.to('cuda:0') # todo

    def batch(self, data):
        maps = data['maps']
        masks = data['masks']
        inp = data['out'] #read in maps, masks, and k-space input

        #initialize z
        total_upsample = 2**(len(self.num_channels_up))
        if total_upsample > 64:
            raise ValueError('desired output size of [320,256] is incompatible with more than 64x upsampling')
        zseed = torch.zeros(self.batch_size,self.num_channels_up[0],320//total_upsample,256//total_upsample)
        if self.use_cpu :
            zseed.data.normal_().type(torch.FloatTensor) #init random input seed
        else:
            zseed.data.normal_().type(torch.cuda.FloatTensor)
            zseed = zseed.to('cuda:0') # todo

        self.z = zseed
        self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.use_sigpy, noncart=self.noncart)
        if self.use_cpu :
            pass
        else:
            self.A = self.A.cuda() # todo

    def forward(self, y):
        out =  self.denoiser(self.z)
        out = out.permute(0,2,3,1)
        print(out.shape)
        return out

    def get_metadata(self):
        return {}


