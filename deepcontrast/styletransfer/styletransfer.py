
#!/usr/bin/env python

import numpy as np
import torch
import sys
import random 

import pytorch_lightning as pl
from torchmetrics.image import TotalVariation
from torchmetrics.image import StructuralSimilarityIndexMeasure

from torchvision import transforms
from torchvision.utils import make_grid
from torch.optim import lr_scheduler 
from dataset import CustomMRIDataset
from utils import stylecontentloss as scloss
import torchvision.models as models


@torch.jit.script


def dot_batch(x1, x2):
# Finds the dot product of two multidimensional Tensors holding batches of data.
    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)
def ip_batch(x):
#Finds the identity product of a multidimensional Tensor holding a batch of data.
    return dot_batch(x, x)
def calc_nrmse(gt, pred):
    return (ip_batch(pred - gt) / ip_batch(gt)).sqrt().mean()

def calc_ssim():
    return StructuralSimilarityIndexMeasure(reduction='elementwise_mean')



class StyleTransfer(pl.LightningModule):
    """StyleTransfer is an abstract class which outlines common functionality for style transfer implementations. 
    All such implementations share hyperparameter initialization, batch data loading, loss function, training step, and optimizer code. 
    Each implementation of StyleTransfer must provide batch, forward, and get_metadata methods in order to define how batches are created 
    from the data, how the model performs its forward pass, and what metadata the user should be able to return. StyleTransfer automatically 
    builds the dataset as an CustomMRIDataSet object; overload _build_data to circumvent this.

    """

    def __init__(self, hparams, trial):
        super(StyleTransfer, self).__init__()
        self.scheduler = None
        self.log_dict = None
        self.trial = trial
        self._init_hparams(hparams) #self.hparams = hparams is depreciated; found this solution at https://github.com/Lightning-AI/lightning/discussions/7525
        self._build_data()

    def _init_hparams(self, hparams):
        # Purpose: initiate the hyperparameters defined in the config file by the user
        self.hparams.update(vars(hparams))
        self._define_loss()
        self._define_reg()
        # you can always design more loss functions below and set them here with various hparam flags and if statements!

    def _define_loss(self):
        # Purpose: define the loss function based on the inputted hyperparameter "which_loss"
        if self.hparams.which_loss=='l1_loss':
            self.loss_fun = torch.nn.L1Loss(reduction='sum')
        elif self.hparams.which_loss=='mse_loss':
            self.loss_fun = torch.nn.MSELoss(reduction='sum')
        elif self.hparams.which_loss=='tv_loss':
            self.loss_fun = TotalVariation(reduction='sum')
        elif self.hparams.which_loss=='ssim_loss':
            # Note: this isn't the final loss, we need to define 1-loss in training step (debugging)
            self.loss_fun = StructuralSimilarityIndexMeasure(reduction='elementwise_mean',kernel_size=5)
        elif self.hparams.which_loss =='style_content_mse':
            self.loss_fun = self._define_style_content_loss
            

    def _define_style_content_loss(self,x_gt,x_hat,x_inp):
        #cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
        with torch.no_grad():
            vgg19 = models.vgg19(pretrained=True).to(x_gt.device)
            cnn = vgg19.features.eval()
            # NOTE: dditionally, VGG networks are trained on images with each channel normalized by 
            # mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]. We will use them to normalize the image before sending it into the network.

            # define the layers for which we want to compute each loss
            content_layers = ['conv_4']
            style_layers= ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
            cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
            cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
        loss = scloss.get_style_content_losses(cnn, cnn_normalization_mean, cnn_normalization_std,
                           x_gt, x_inp, x_hat,
                           content_layers,
                           style_layers, self.hparams.style_weight,self.hparams.content_weight)
        loss_mse = torch.nn.MSELoss(reduction='sum')
        return loss_mse(x_gt,x_hat) + loss


    def _define_reg(self):
        # Purpose: create a regularization term, scaled by reg rate lambda
        if self.hparams.which_reg == 'tv_reg':
            self.reg = TotalVariation(reduction='sum')
        elif self.hparams.which_reg == 'l1_reg':
            self.reg = torch.nn.L1Loss(reduction='sum')
        elif self.hparams.which_reg == 'ssim_reg':
            # Note: this isn't the final reg term, we need to define 1-loss in training step (debugging)
            self.reg = StructuralSimilarityIndexMeasure(reduction='elementwise_mean',kernel_size=5)
        else:
            self.reg = None

        
    def forward(self, y):

        """Not implemented, defined in classes that inherit from StyleTransfer.
        """

    def batch(self, data):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError

    def _build_data(self):
        # Purpose: Make the dataset objects for training, validation, and testing
        # NOTE, for now, the default transform is ToTensor. Consider making this configurable through hparam in the future
        # NOTE, consider switching to using a sampler instead of shuffle=True
        #transform_final = []
        #if self.hparams.data_augment:
        #    rand_ang = random.randrange(-90,90)
        #    # this doesn't treat inp and gt the same; does different rotation for each :(
        #    #transform_final = transforms.Compose([transforms.ToTensor(), transforms.RandomAffine(90, translate=None, scale=None, shear=None, resample=0, fillcolor=0)])
        #    transform_final = transforms.Compose([transforms.ToTensor(), transforms.functional.rotate(angle=rand_ang)])
        #else:
        #transform_final = transforms.ToTensor()
        self.D_train = CustomMRIDataset(self.hparams.img_dirs_train, self.hparams.target_size, limit_range = self.hparams.limit_data_range, transform=transforms.ToTensor())
        self.D_val = CustomMRIDataset(self.hparams.img_dirs_val, self.hparams.target_size, limit_range = self.hparams.limit_data_range,transform=transforms.ToTensor())
        self.D_test = CustomMRIDataset(self.hparams.img_dirs_test, self.hparams.target_size, limit_range = self.hparams.limit_data_range, transform=transforms.ToTensor())
    
    def train_dataloader(self):
        # Purpose: create a training dataloader to pull batches for training
        # Note: It's good to have shuffle ON or to use a sampler because you don't want the network to learn based on the order of the data
        return torch.utils.data.DataLoader(self.D_train, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        # Purpose: create a validation dataloader for the validation step
        # pytorch strongly recommends during shuffle off for validation dataset...
        return torch.utils.data.DataLoader(self.D_val, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=True)

    def test_dataloader(self):

        # Purpose: create a testing dataloader for testing
        return torch.utils.data.DataLoader(self.D_test, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=True)

    def training_step(self, train_batch, batch_idx):
        # Purpose: Define the training step that pytorch lightning calls for training
        # Pulls a batch, computes the loss, and outputs metrics and images to tensorboard
        # Returns: loss

        # ground truth (nfs) and input (fs) pairs from batch
        gt, inp = train_batch

        if self.hparams.data_augment:
            # rotate the images by the same random angle if data_augment is true
            rand_ang = random.randrange(-10,10)
            gt = transforms.functional.rotate(gt, angle=rand_ang)
            inp = transforms.functional.rotate(inp, angle=rand_ang)

        # pass input through forward to get prediction, x_hat
        x_hat = self.forward(inp)


        # FOR SANITY CHECKING:
        print("gt.max = ", torch.max(gt), "gt.min = ", torch.min(gt))
        print("inp.max = ", torch.max(inp), "inp.min = ", torch.min(inp))
        print("x_hat.max = ", torch.max(x_hat), "x_hat.min = ", torch.min(x_hat))


        # everything under this if statement configures the images for viewing in tensorboard through logger
        if self.logger:
            _x_hat=0
            _x_gt=0
            _inp = 0
            if self.hparams.accelerator == 'gpu':
               _x_hat = x_hat.detach().cpu().numpy() 
               _x_gt = gt.detach().cpu().numpy()
               _inp = inp.detach().cpu().numpy()
            else:
               _x_hat = x_hat.detach().numpy() 
               _x_gt = gt.detach().numpy()
               _inp = inp.detach().numpy()

            # save the images at save_out_path if the save_img flag is toggled to True
            if self.hparams.save_img:
                if ((self.current_epoch % self.hparams.save_every_N_steps == 0)) or (self.current_epoch == self.hparams.num_epochs - 1):
                    np.save("{0}/{1}_version{2}_epoch{3}_x_gt.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _x_gt)
                    np.save("{0}/{1}_version{2}_epoch{3}_x_hat.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _x_hat)
                    np.save("{0}/{1}_version{2}_epoch{3}_x_inp.npy".format(self.hparams.save_img_path, self.hparams.tt_logger_name, self.hparams.tt_logger_version, self.current_epoch), _inp)

            # If your batch is less than 3, you will need to configure logger to read in less images from the batch (by default it reads 3)
            if self.hparams.batch_size >= 3:

                # add x_hat (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_hat[0,...], _x_hat[1,...], _x_hat[2,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=3, pad_value=10)
                self.logger.experiment.add_image('5_c_train_x_hat', grid, self.current_epoch)

                # add x_gt (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_gt[0,...], _x_gt[1,...], _x_gt[2,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=3, pad_value=10)
                self.logger.experiment.add_image('5_a_ground_truth', grid, self.current_epoch)

                # add input (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_inp[0,...], _inp[1,...], _inp[2,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=3, pad_value=10)
                self.logger.experiment.add_image('5_b_input', grid, self.current_epoch)

            elif self.hparams.batch_size == 2:

                # add x_hat (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_hat[0,...], _x_hat[1,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=2, pad_value=10)
                self.logger.experiment.add_image('5_c_train_x_hat', grid, self.current_epoch)

                # add x_gt (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_gt[0,...], _x_gt[1,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=2, pad_value=10)
                self.logger.experiment.add_image('5_a_ground_truth', grid, self.current_epoch)

                # add input (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_inp[0,...], _inp[1,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=2, pad_value=10)
                self.logger.experiment.add_image('5_b_input', grid, self.current_epoch)

            elif self.hparams.batch_size == 1:
                myim = torch.tensor(_x_hat)
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1, pad_value=10)
                self.logger.experiment.add_image('5_c_train_x_hat', grid, self.current_epoch)

                # add x_gt (first 3 in batch) to the grid
                myim = torch.tensor(_x_gt)
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1, pad_value=10)
                self.logger.experiment.add_image('5_a_ground_truth', grid, self.current_epoch)

                # add input (first 3 in batch) to the grid
                myim = torch.tensor(_inp)
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1, pad_value=10)
                self.logger.experiment.add_image('5_b_input', grid, self.current_epoch)



        # define the loss and get plotable loss for tensorboard
        loss = []
        if self.hparams.which_loss == 'ssim_loss':
            loss = 1.-self.loss_fun(gt, x_hat)
        elif self.hparams.which_loss=='style_content_mse':
            loss = self.loss_fun(gt,x_hat,inp)
        else:
            loss = self.loss_fun(gt,x_hat)


        if self.reg is not None: #if there is a regularization term...
            if self.hparams.which_reg == 'l1_reg':
                loss = self.loss_fun(x_hat, gt)+self.hparams.lamb*self.reg(x_hat, gt)
            elif self.hparams.which_reg == 'ssim_reg':
                loss = self.loss_fun(x_hat, gt)+self.hparams.lamb*(1.-self.reg(gt,x_hat))
            elif self.hparams.which_reg == 'tv_reg':
                loss = self.loss_fun(x_hat, gt)+self.hparams.lamb*self.reg(x_hat)
        
        #scalar loss for plotting
        _loss = loss.clone().detach().requires_grad_(False)

        # get other scalar metrics for reporting in tensorboard
        _epoch = self.current_epoch
        _nrmse = calc_nrmse(gt, x_hat).detach().requires_grad_(False)
        ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean',kernel_size=5).to(gt.device)
        ssim_gt = ssim(gt, gt)
        _ssim = ssim(gt, x_hat).detach().requires_grad_(False)
        
        # making sure SSIM is computing what it's supposed to (this should be 1)
        if self.current_epoch==0:
            print("SSIM sanity check is...", ssim_gt)

        log_dict = {
                '0_epoch': self.current_epoch,
                '1_train_loss': _loss,
                '2_nrmse': _nrmse,
                '3_ssim': _ssim,
                }

        # update logger with log_dict keys!
        if self.logger:
            for key in log_dict.keys():
                self.logger.experiment.add_scalar(key, log_dict[key], self.global_step)

        self.log_dict = log_dict

        return loss

    def validation_step(self, val_batch, batch_idx):
        # Purpose: define the validation step that Pytorch Lightning calls for validation
        # Same structure as training_step(*) but uses batches from validation dataloader

        # ground truth and input pairs from batch
        gt, inp = val_batch
        # pass input through forward to get prediction, x_hat
        x_hat = self.forward(inp)
        
        if self.logger:
            _x_hat=0
            _x_gt=0
            _inp = 0
            if self.hparams.accelerator == 'gpu':
               _x_hat = x_hat.detach().cpu().numpy() 
               _x_gt = gt.detach().cpu().numpy()
               _inp = inp.detach().cpu().numpy()
            else:
               _x_hat = x_hat.detach().numpy() 
               _x_gt = gt.detach().numpy()
               _inp = inp.detach().numpy()

            if self.hparams.batch_size >= 3:

                # add x_hat (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_hat[0,...], _x_hat[1,...], _x_hat[2,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=3, pad_value=10)
                self.logger.experiment.add_image('5_f_val_x_hat', grid, self.current_epoch)

                # add x_gt (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_gt[0,...], _x_gt[1,...], _x_gt[2,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=3, pad_value=10)
                self.logger.experiment.add_image('5_d_ground_truth', grid, self.current_epoch)

                # add input (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_inp[0,...], _inp[1,...], _inp[2,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=3, pad_value=10)
                self.logger.experiment.add_image('5_e_input', grid, self.current_epoch)
            
            elif self.hparams.batch_size == 2:
                # add x_hat (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_hat[0,...], _x_hat[1,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=2, pad_value=10)
                self.logger.experiment.add_image('5_f_val_x_hat', grid, self.current_epoch)

                # add x_gt (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_x_gt[0,...], _x_gt[1,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=2, pad_value=10)
                self.logger.experiment.add_image('5_d_ground_truth', grid, self.current_epoch)

                # add input (first 3 in batch) to the grid
                myim = torch.tensor(np.stack((_inp[0,...], _inp[1,...]), axis=0))
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=2, pad_value=10)
                self.logger.experiment.add_image('5_e_input', grid, self.current_epoch)

            elif self.hparams.batch_size == 1:
                # add x_hat (first 3 in batch) to the grid
                myim = torch.tensor(_x_hat)
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1, pad_value=10)
                self.logger.experiment.add_image('5_f_val_x_hat', grid, self.current_epoch)

                # add x_gt (first 3 in batch) to the grid
                myim = torch.tensor(_x_gt)
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1, pad_value=10)
                self.logger.experiment.add_image('5_d_ground_truth', grid, self.current_epoch)

                # add input (first 3 in batch) to the grid
                myim = torch.tensor(_inp)
                grid = make_grid(myim, scale_each=True, normalize=True, nrow=1, pad_value=10)
                self.logger.experiment.add_image('5_e_input', grid, self.current_epoch)


        # compute the validation loss
        val_loss = []
        if self.hparams.which_loss == 'ssim_loss':
            val_loss = 1.-self.loss_fun(gt, x_hat)
        elif self.hparams.which_loss=='style_content_mse':
            val_loss = self.loss_fun(gt,x_hat,inp)
        else:
            val_loss = self.loss_fun(gt,x_hat)

        if self.reg is not None: #if there is a regularization term...
            if self.hparams.which_reg == 'l1_reg':
                val_loss = self.loss_fun(x_hat, gt)+self.hparams.lamb*self.reg(x_hat, gt)
            elif self.hparams.which_reg == 'ssim_reg':
                loss = self.loss_fun(x_hat, gt)+self.hparams.lamb*(1.-self.reg(gt,x_hat))
            elif self.hparams.which_reg == 'tv_reg':
                val_loss = self.loss_fun(x_hat, gt)+self.hparams.lamb*self.reg(x_hat)
        
        _val_loss = val_loss.clone().detach().requires_grad_(False)
        _val_nrmse = calc_nrmse(gt, x_hat).detach().requires_grad_(False)

        # Get the SSIM
        ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean',kernel_size=5).to(gt.device)
        ssim_gt = ssim(gt, gt)
        _val_ssim = ssim(gt, x_hat).detach().requires_grad_(False)
        # Sanity check that SSIM is computing what it's supposed to again...
        if self.current_epoch==0:
            print("SSIM sanity check is...", ssim_gt)


        # Update log dict by adding on validation loss
        log_dict = {
                '1_val_loss': _val_loss,
                '2_val_nrmse': _val_nrmse,
                '3_val_ssim': _val_ssim
                }

        # update logger with log_dict keys!
        if self.logger:
            for key in log_dict.keys():
                self.logger.experiment.add_scalar(key, log_dict[key], self.global_step)

        self.log_dict_val = log_dict
        #return val_loss

    def test_step(self, test_batch, batch_idx):
        self.validation_step(test_batch, batch_idx)

    def get_progress_bar_dict(self):
        items = super(StyleTransfer, self).get_progress_bar_dict()
        if self.log_dict:
            for key in self.log_dict.keys():
                if type(self.log_dict[key]) == torch.Tensor:
                    items[key] = utils.itemize(self.log_dict[key])
                else:
                    items[key] = self.log_dict[key]
        return items

    def configure_optimizers(self):
        """Determines whether to use Adam or SGD depending on hyperparameters.
        Returns:
            Torch’s implementation of SGD or Adam, depending on hyperparameters.
        """

        if 'adam' in self.hparams.solver:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.step,betas=(self.hparams.adam_beta1, self.hparams.adam_beta2),eps=self.hparams.adam_eps,weight_decay=self.hparams.weight_decay)
        elif 'sgd' in self.hparams.solver:
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.step)
        elif 'rmsprop' in self.hparams.solver:
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.step) 
        if(self.hparams.lr_scheduler != -1):
            # doing self.scheduler will create a scheduler instance in our self object
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.hparams.lr_scheduler[0], gamma=self.hparams.lr_scheduler[1])
        if self.scheduler is None:
            return [self.optimizer]
        else:                
            return [self.optimizer], [self.scheduler]

    def on_after_backward(self):
        _grad_norms_all = pl.utilities.grad_norm(self.network,2)
        _grad_norm_total = _grad_norms_all['grad_2.0_norm_total']
        #for k in _grad_norms_all.keys():
        #    print("key: ",k, "   item: ", _grad_norms_all[k])
        self.logger.experiment.add_scalar('_grad_norm_total', _grad_norm_total, self.global_step)
        



