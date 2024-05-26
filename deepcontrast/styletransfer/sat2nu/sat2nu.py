#!/usr/bin/env python


from styletransfer import StyleTransfer
from models import UNet
import torch
import numpy as np

class Sat2Nu(StyleTransfer):

    def __init__(self, args, trial):
        super(Sat2Nu, self).__init__(args,trial)

        self.network = UNet(dropout=self.hparams.dropout, shallow_net=self.hparams.shallow_net)
   
    def forward(self, x): # |y - Ax|
        out_net =  self.network(x)
        #print("-------> out_net.shape = ", out_net.shape)
        self.out_net = out_net

        # save the model at PATH on either {save_every_N_epochs} or on the very last epoch
        if self.hparams.save_model and (self.current_epoch == self.hparams.num_epochs - 1):
            PATH = "{0}/{1}_version{2}_epoch{3}_state_dict.pt".format(self.hparams.save_out_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch)
            #if (self.current_epoch % self.hparams.save_every_N_steps == 0) or (self.current_epoch == self.hparams.num_epochs - 1):
            torch.save(self.network.state_dict(), PATH)
        return out_net

    def get_metadata(self):
        return {}

    # for style transfer, I think we will have to write a custom training_step() function here that overrides the one from the parent class
