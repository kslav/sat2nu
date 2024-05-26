#!/usr/bin/env python

import torch
from styletransfer import Sat2Nu
from models import UNet
from dataset import CustomMRIDataset
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# define some useful functions

def dot_batch(x1, x2):
# Finds the dot product of two multidimensional Tensors holding batches of data.
    batch = x1.shape[0]
    return torch.reshape(x1*x2, (batch, -1)).sum(1)
def ip_batch(x):
#Finds the identity product of a multidimensional Tensor holding a batch of data.
    return dot_batch(x, x)
def calc_nrmse(gt, pred):
    return (ip_batch(pred - gt) / ip_batch(gt)).sqrt().mean()


# define the directories for the model and dataset
model_dir = '/cbica/home/slavkovk/project_DeepContrast/logs/models_saved/sat2nu_lr_001_version5_epoch1000_state_dict.pt'
img_dirs = "/cbica/home/slavkovk/project_DeepContrast/Data/basser_val_dirs_2D_sag_shuffled_DEBUG.csv"

# load the model
print("Loading the model state...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
model.load_state_dict(torch.load(model_dir))

# establish data and dataloader
print("Creating data loader...")
batch_num = 48
D_train = CustomMRIDataset(img_dirs, 256, transform=transforms.ToTensor())
train_dataloader = torch.utils.data.DataLoader(D_train, batch_size=batch_num, shuffle=False, num_workers=1, drop_last=True)

# run inference on first image in batch as a test
print("Loading a batch of data...")
x_gt, x_inp = next(iter(train_dataloader))

for i in range(0,batch_num):
    x_inp_i = x_inp[i,...]
    x_gt_i = x_gt[i,...]
    
    x_inp_i = x_inp_i[None,...].to(device)
    x_gt_i = x_gt_i[None,...].to(device)
    print("Running inference...")
    x_hat = model(x_inp_i)

    # compute metrics
    print("Computing metrics...")
    _val_nrmse = calc_nrmse(x_gt_i, x_hat)
    ssim = StructuralSimilarityIndexMeasure(reduction='elementwise_mean',kernel_size=5).to(device)
    _val_ssim = ssim(x_gt_i, x_hat)
    print("_val_nrmse = ", _val_nrmse)
    print("val_ssim", _val_ssim)

