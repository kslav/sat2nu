#!/usr/bin/env python


from test_tube import HyperOptArgumentParser
from pytorch_lightning import Trainer
#from pytorch_lightning.loggers import TestTubeLogger #deprecated
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
#

from pytorch_lightning import seed_everything

import os
import pathlib
import argparse

import time

from styletransfer import Sat2Nu

import torch
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

import random # used to avoid race conditions, intentionall unseeded
import numpy.random

def main_train(args, gpu_ids=None):

    if args.hyperopt:
        time.sleep(random.random()) # used to avoid race conditions with parallel jobs
    tt_logger = TensorBoardLogger(save_dir=args.logdir, name=args.name, version=args.version)
    tt_logger.log_hyperparams(args)
    save_path = '{}/{}/version_{}'.format(args.logdir, tt_logger.name, tt_logger.version)
    args.save_path = save_path
    args.tt_logger_name = tt_logger.name
    args.tt_logger_version = tt_logger.version
    checkpoint_path = '{}/checkpoints'.format(save_path)
    pathlib.Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    if args.save_all_checkpoints:
        save_top_k = -1
    else:
        save_top_k = 1
    checkpoint_callback = ModelCheckpoint(checkpoint_path, 'epoch', save_top_k=save_top_k, mode='max', verbose=False)

    if args.styletransfer == 'sat2nu':
        MyST = Sat2Nu
    else:
        raise NotImplementedError

    M = MyST(args)

    if args.checkpoint_init:
        # FIXME: workaround for PyTL issues with loading hparams
        print('loading checkpoint: {}'.format(args.checkpoint_init))
        checkpoint = torch.load(args.checkpoint_init, map_location=lambda storage, loc: storage)
        M.load_state_dict(checkpoint['state_dict'])
    else:
        print('training from scratch')

    
    if gpu_ids is None:
        gpus = None
        distributed_backend = None # FIXME should this also be ddp?
    elif args.hyperopt:
        gpus = 1
        distributed_backend = None
    else:
        gpus = gpu_ids
        distributed_backend = 'ddp'

    if args.accelerator =="gpu":
        print("Defining trainer with GPU accelerators")
        trainer = Trainer(max_epochs=args.num_epochs, accelerator=args.accelerator, devices=gpu_ids, logger=tt_logger, callbacks=checkpoint_callback, accumulate_grad_batches=args.num_accumulate, gradient_clip_val=args.clip_grads, log_every_n_steps=args.save_every_N_steps)
    elif args.accelerator == "cpu":
        print("Defining trainer with CPU accelerators")
        trainer = Trainer(max_epochs=args.num_epochs, accelerator=args.accelerator, devices=args.num_workers, logger=tt_logger, callbacks=checkpoint_callback, accumulate_grad_batches=args.num_accumulate, gradient_clip_val=args.clip_grads, log_every_n_steps=args.save_every_N_steps)

    trainer.fit(M) # training and validation steps!
    trainer.test(M) # test on in-distribution data if you choose

if __name__ == '__main__':
    usage_str = 'usage: %(prog)s [options]'
    description_str = 'style transfer for MRI'

    parser = HyperOptArgumentParser(usage=usage_str, description=description_str, formatter_class=argparse.ArgumentDefaultsHelpFormatter, strategy='grid_search')

    parser.opt_list('--step', action='store', dest='step', type=float, tunable=False, options=[0.001,0.0001], help='number of latent channels', default=0.0001)
    
    parser.opt_list('--solver', action='store', dest='solver', type=str, tunable=False, options=['sgd', 'adam', 'rmsprop'], help='optimizer/solver ("adam", "sgd", etc.)', default="adam")
    parser.opt_range('--batch_size', action='store', dest='batch_size', type=int, tunable=False, low=1, high=20, help='batch size', default=10)
    parser.opt_range('--num_admm', action='store', dest='num_admm', type=int, tunable=False, low=1, high=10, nb_samples=4, help='number of ADMM iterations', default=3)
    parser.opt_list('--network', action='store', dest='network', type=str, tunable=False, options=['UNet'], help='which model to use', default='UNet')
    parser.opt_range('--dropout', action='store', dest='dropout', type=float, tunable=False, low=0., high=.5, help='dropout fraction', default=0.)
    parser.opt_list('--batch_norm', action='store_true', dest='batch_norm', tunable=False, options=[True, False], help='batch normalization', default=False)
    parser.opt_range('--clip_grads', action='store', type=float, dest='clip_grads', help='clip norm of gradient vector to val', default=None, tunable=False, nb_samples=10, low=0, high=500)

    parser.add_argument('--num_accumulate', action='store', dest='num_accumulate', type=int, help='nunumber of batch accumulations', default=1)
    parser.add_argument('--target_size',action='store',dest='target_size',type=float, help='size you want to resize input image to',default=256)
    parser.add_argument('--name', action='store', dest='name', type=str, help='experiment name', default=1)
    parser.add_argument('--accelerator', action='store', dest='accelerator', type=str, help='type of accelerator (cpu, gpu, etc)', default="cpu")
    parser.add_argument('--version', action='store', dest='version', type=int, help='version number', default=None)
    parser.add_argument('--gpu', action='store', dest='gpu', type=str, help='gpu IDs', default=None)
    parser.add_argument('--num_epochs', action='store', dest='num_epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--random_seed', action='store', dest='random_seed', type=int, help='random number seed for numpy', default=723)
    parser.add_argument('--styletransfer', action='store', type=str, dest='styletransfer', default='sat2nu', help='reconstruction method')
    parser.add_argument('--img_dirs_train', action='store', dest='img_dirs_train', type=str, help='training data directory', default=None)
    parser.add_argument('--img_dirs_val', action='store', dest='img_dirs_val', type=str, help='validation data directory', default=None)
    parser.add_argument('--img_dirs_test', action='store', dest='img_dirs_val', type=str, help='test data directory', default=None)
    parser.add_argument('--num_workers', action='store', type=int,  dest='num_workers', help='number of workers', default=1)
    parser.add_argument('--shuffle', action='store_true', dest='shuffle', help='shuffle input data files each epoch', default=False)
    parser.add_argument('--max_norm_constraint', action='store', type=float, dest='max_norm_constraint', help='norm constraint on weights', default=None)
    parser.add_argument('--adam_eps', action='store', type=float, dest='adam_eps', help='adam epsilon', default=1e-8)
    parser.add_argument('--adam_beta1', action='store', type=float, dest='adam_beta1', help='adam beta1', default=0.9)
    parser.add_argument('--adam_beta2', action='store', type=float, dest='adam_beta2', help='adam beta2', default=0.999)
    parser.add_argument('--weight_decay', action='store', type=float, dest='weight_decay', help='weight decay param in (adam) optimizer', default=0)
    parser.add_argument('--self_supervised', action='store_true', dest='self_supervised', help='self-supervised loss', default=False)
    parser.add_argument('--hyperopt', action='store_true', dest='hyperopt', help='perform hyperparam optimization', default=False)
    parser.add_argument('--checkpoint_init', action='store', dest='checkpoint_init', type=str, help='load from checkpoint', default=None)
    parser.add_argument('--logdir', action='store', dest='logdir', type=str, help='log dir', default='logs')
    parser.add_argument('--save_all_checkpoints', action='store_true', dest='save_all_checkpoints', help='Save all checkpoints', default=False)
    parser.add_argument('--lr_scheduler', action='store', dest='lr_scheduler', nargs='+', type=int, help='do [#epoch, learning rate multiplicative factor] to use a learning rate scheduler', default=-1)
    parser.add_argument('--save_every_N_steps', action='store', type=int,  dest='save_every_N_steps', help='save images every N epochs', default=1)
    parser.add_argument('--do_warmstart', action='store_true', dest='do_warmstart', help='load preliminary weights rather than random ones', default=False)
    parser.add_argument('--data_augment', action='store_true', dest='data_augment', help='use random rotations in training data loader', default=False)
    parser.add_argument('--lamb', action='store', type=int,  dest='lamb', help='regularization rate if regularization used', default=0.00001)
    parser.add_argument('--save_img', action='store_true', dest='save_img', help='toggle on/off whether to save outputted images', default=False)
    parser.add_argument('--save_model', action='store_true', dest='save_model', help='toggle on/off whether to save the model', default=False)
    parser.add_argument('--limit_data_range', action='store_true', dest='limit_data_range', help='limit data range to [0, 1]', default=False)
    parser.json_config('--config', default=None)
    parser.add_argument('--save_out_path', action='store', dest='save_out_path', type=str, help='points to where to save images and models. default is home dir', default='/cbica/home/slavkovk')
    parser.add_argument('--shallow_net', action='store', dest='shallow_net', type=str, help='toggles on/off whether to use a shallow version of a network', default=False)
    parser.add_argument('--which_loss', action='store', dest='which_loss', type=str, help='specify which loss fun you want to use', default='mse_loss')
    parser.add_argument('--which_reg', action='store', dest='which_reg', type=str, help='specify which regularization you want to use', default='None')



    args = parser.parse_args()


    torch.manual_seed(args.random_seed)
    numpy.random.seed(args.random_seed)
    torch.backends.cudnn.deterministic=True
    seed_everything(args.random_seed)

    if args.hyperopt:
        if args.gpu is None:
            args.accelerator = "cpu"
            args.optimize_parallel_cpu(main_train, nb_trials=args.num_trials, nb_workers=args.num_workers)
        else:
            gpu_ids = [a.strip() for a in args.gpu.split(',')]
            args.optimize_parallel_gpu(main_train, gpu_ids=gpu_ids, max_nb_trials=args.num_trials)
    else:
        if args.gpu is not None:
            gpu_ids = [int(a) for a in args.gpu.split(',')] # GPU IDs inputted as list
        else:
            gpu_ids = None
        main_train(args, gpu_ids=gpu_ids)
