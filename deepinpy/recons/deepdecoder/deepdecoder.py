from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import DeepDecoder
from deepinpy.recons import Recon
import torch

class DeepDecoderRecon(Recon):

	def __init__(self, args):
        self.num_channels_up = [128]*6
		self.x_adj = None
		#output_size = self.A.img_shape
		super(CSDIPRecon, self).__init__(args)
		# self.denoiser = DeepDecoder(ngf=64, output_size=[320, 256], nc=2, num_measurements=256)
        self.denoiser = DeepDecoder(num_output_channels=2,num_channels_up=self.num_channels_up, upsample_first=True, need_sigmoid=False) #todo: check whether I need to use sigmoid
        if self.cpu:
            pass
        else:
            self.denoiser = self.denoiser.type(torch.cuda.FloatTensor)
	def batch(self, data):
		maps = data['maps']
		masks = data['masks']
		inp = data['out'] #read in maps, masks, and k-space input

        #initialize z
        total_upsample = 2**(len(self.num_channels_up))
        if total_upsample > 64:
            raise ValueError('desired output size of [320,256] is incompatible with more than 64x upsampling')
		zseed = torch.zeros(self.batch_size,self.num_channels_up[0],320//total_upsample,256//total_upsample)
		if self.cpu == True:
			zseed.data.normal_().type(torch.FloatTensor) #init random input seed
		else:
			zseed.data.normal_().type(torch.cuda.FloatTensor)
			zseed = zseed.to('cuda:0')

		self.z = zseed
		self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.use_sigpy, noncart=self.noncart)

	def forward(self, y):
		out =  self.denoiser(self.z)
		return out

	def get_metadata(self):
		return {}


