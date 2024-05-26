import sys

from models import convdecoder
from styletransfer import StyleTransfer
import numpy as np
import torch


class Sat2Convu(StyleTransfer):

    def __init__(self, args, trial):
        super(Sat2Convu, self).__init__(args,trial)

        self.N1 = 8
        self.N2 = 8
        self.x_adj = None

        self.output_size = [self.hparams.target_size, self.hparams.target_size]#self.D.shape[1:]
        print('output size:', self.output_size)
        self.hparams.num_image_params = np.product(self.output_size) #* 2 # real/imaginary

        #if len(self.output_size) > 2:
        #    self.num_output_channels = 2 * np.prod(self.output_size[:-2])
        #    self.output_size = self.output_size[-2:]
        #else:
        #    self.num_output_channels = 2
        self.num_output_channels = 1

        if self.hparams.network == 'ConvDecoder':
            
            self.N1, self.N2 = 16, 16
            self.in_size = [self.N1, self.N2]

            self.network = convdecoder(num_output_channels=self.num_output_channels, strides=[1]*self.hparams.num_blocks, out_size=self.output_size, in_size=self.in_size, num_channels=self.hparams.latent_channels, z_dim=self.hparams.z_dim, num_layers=self.hparams.num_blocks, need_sigmoid=False, upsample_mode='nearest', skips=False, need_last=True)

        else:
            # FIXME: error logging
            print('ERROR: invalid network specified')
            sys.exit(-1)
        
        # self.network gets defined above, so at this point you can do self.network.load_state_dict(torch.load(PATH),strict=False).
        # Need an if-statement based on a flag that says "yes warmstart"
        if self.hparams.do_warmstart:
            self.network.load_state_dict(torch.load(self.hparams.state_dict_path),strict=False)
        #self.zseed = None # to be initialized on first batch

        # save number of image and model params to the hyperparameters struct
        self.hparams.num_network_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        self.hparams.compression_factor = self.hparams.num_image_params / self.hparams.num_network_params

        #self.optimize_z = self.hparams.optimize_z
        print("----->len(self.D_train)", len(self.D_train))
        #zseed = torch.zeros(len(self.D), self.hparams.z_dim, self.N1, self.N2) 
        zseed = torch.zeros(self.hparams.z_dim, self.N1, self.N2) 

        zseed.data.normal_().type(torch.FloatTensor)
        print("-------> from init, zseed.shape = ", zseed.shape)
        self.zseed = zseed


        #if self.optimize_z:
        #    self.zseed = torch.nn.Parameter(self.zseed)

    #def batch(self, data):
    #    maps = data['maps']
    #    masks = data['masks']
    #    inp = data['out'] #read in maps, masks, and k-space input#

        # initialize z vector only once per index
    #    if not self.optimize_z:
    #        self.zseed = self.zseed.to(inp.device)

    #    self.A = MultiChannelMRI(maps, masks, l2lam=0.,  img_shape=data['imgs'].shape, use_sigpy=self.hparams.use_sigpy, noncart=self.hparams.noncart)

    #   self.batch_idx = data['idx']

    def forward(self, x):
        zseed = self.zseed#[self.hparams.batch_size,...]
        print("-------> from forward, zseed.shape = ", zseed[None,...].shape)
        if self.hparams.batch_size == 1:
            zseed = zseed[None,...]
        out =  self.network(zseed.to(x.device)) #DCGAN acts on the low-dim space parameterized by z to output the image x
        return out

    def get_metadata(self):
        return {}
