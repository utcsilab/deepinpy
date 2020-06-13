from deepinpy.forwards import MultiChannelMRI
from deepinpy.models import DeepDecoder
from deepinpy.recons import Recon
import torch


class CSDIPRecon(Recon):

	def __init__(self, args):
		self.Z_DIM = 64
		self.x_adj = None
		#output_size = self.A.img_shape
		super(CSDIPRecon, self).__init__(args)
		self.denoiser = DCGAN_MRI(self.Z_DIM, ngf=64, output_size=[320, 256], nc=2, num_measurements=256)
	def batch(self, data):
		maps = data['maps']
		masks = data['masks']
		inp = data['out'] #read in maps, masks, and k-space input

        #initialize z

		zseed = torch.zeros(self.batch_size*self.Z_DIM).view(self.batch_size,self.Z_DIM,1,1)
		if self.cpu == True:
			#print('Im in CPU!')
			zseed.data.normal_().type(torch.FloatTensor) #init random input seed
		else:
			#print('Im in GPU!')
			zseed.data.normal_().type(torch.cuda.FloatTensor)
			zseed = zseed.to('cuda:0')
			#print('kkkkkkkkkkk',[self.denoiser.parameters()][0].device)

		self.z = zseed
		#print('jjjjjjjjjjjjjjjjjjj', self.z.device)
		self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.use_sigpy, noncart=self.noncart)
        #self.x_adj = self.A.adjoint(inp)

	def forward(self, y):
		out =  self.denoiser(self.z) #DCGAN acts on the low-dim space parameterized by z to output the image x
		return out

	def get_metadata(self):
		return {}


#Gotta be careful with the set of arguments for the DCGAN. In the original code, we have the following:
#args.Z_DIM, NGF, args.IMG_SIZE,\
#            args.NUM_CHANNELS, args.NUM_MEASUREMENTS

#data['imgs'].shape

