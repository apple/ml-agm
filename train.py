#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import random
import argparse

import copy
from pathlib import Path
import numpy as np
import torch
from torch.multiprocessing import Process

from edm.logger import Logger
from edm.distributed_util import init_processes
from dataset import spiral,cifar10, AFHQv2,imagenet64
from AM import runner
from configs import cifar10_config,toy_config,afhqv2_config,imagenet64_config
import colored_traceback.always
import torch.multiprocessing

RESULT_DIR = Path("results")
def set_seed(seed):
    # https://github.com/pytorch/pytorch/issues/7068
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # https://stackoverflow.com/questions/73125231/pytorch-dataloaders-bad-file-descriptor-and-eof-for-workers0
    torch.multiprocessing.set_sharing_strategy('file_system')

def create_training_options():
    # --------------- basic ---------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--name",           type=str,   default=None,                                       help="experiment ID")
    parser.add_argument("--exp",            type=str,   default='toy',  choices=['toy','cifar10','AFHQv2','imagenet64','cond-imagenet64'],          help="experiment type")
    parser.add_argument("--toy-exp",         type=str,  default='gmm',  choices=['gmm','spiral'],          help="experiment type")
    parser.add_argument("--ckpt",           type=str,   default=None,                                       help="resumed checkpoint name")
    parser.add_argument("--cond",           action="store_true",                                            help="whether or not use class cond")
    parser.add_argument("--gpu",            type=int,   default=None,                                       help="set only if you wish to run on a particular device")
    parser.add_argument("--n-gpu-per-node", type=int,   default=1,                                          help="number of gpu on each node")
    parser.add_argument("--master-address", type=str,   default='localhost',                                help="address for master")
    parser.add_argument("--node-rank",      type=int,   default=0,                                          help="the index of node")
    parser.add_argument("--num-proc-node",  type=int,   default=1,                                          help="The number of nodes in multi node env")
    parser.add_argument("--port",           type=str,   default='6022',                                     help="localhost port")

    # --------------- Dynamics Hyperparameters ---------------
    parser.add_argument("--n-train",        type=int,   default=5000)
    parser.add_argument("--t0",             type=float, default=1e-4,                                       help="Number of Training sample for toy dataset")
    parser.add_argument("--T",              type=float, default=0.999,                                      help="Terminal Time for the dynamics")
    parser.add_argument("--nfe",            type=int,   default=1000,                                       help="number of interval")
    parser.add_argument("--varx",           type=float, default=1.0,                                        help="variance of position for prior")
    parser.add_argument("--varv",           type=float, default=1.0,                                        help="variance of velocity for prior")
    parser.add_argument("--k",              type=float, default=0.0,                                          help="Correlation/Covariance of position of velocity for prior distribution")
    parser.add_argument("--p",              type=float, default=3,                                          help="diffusion coefficient for Time Variant value g(t)=p*(damp_t-t)")
    parser.add_argument("--damp-t",         type=float, default=1,                                          help="diffusion coefficient for Time Variant value g(t)=p*(damp_t-t)")
    parser.add_argument("--DE-type",        type=str,   default='probODE', choices=['probODE','SDE'],\
                                                                                                            help="Choose the type of SDE, which includes Time Varing g Model Predictive Control (TVgMPC) ,\
                                                                                                                Time Invariant g Model Predictive Control (TIVgMPC),Time Invariant g Flow Matching (TIVgFM)")
    # --------------- optimizer and loss ---------------
    parser.add_argument("--microbatch",     type=int,   default=512,                                        help="mini batch size for gradient descent")
    parser.add_argument("--num-itr",        type=int,   default=50000,                                      help="number of training iteration")
    parser.add_argument("--lr",             type=float, default=1e-3,                                       help="learning rate")
    parser.add_argument("--ema",            type=float, default=0.9999,                                     help='ema decay rate')
    parser.add_argument("--l2-norm",        type=float, default=0,                                          help='L2 norm for optimizer')
    parser.add_argument("--t-samp",         type=str, default='uniform', choices=['uniform','debug'],\
                                                                                                           help="the way to sample t during sampling")
    parser.add_argument("--precond",        action="store_true",                                            help="preconditioning for the network output")
    parser.add_argument("--clip-grad",      type=float, default=None,                                       help="whether to clip the gradient.")
    parser.add_argument("--xflip",          action="store_true",                                            help="Whether flip the dataset in x-horizon")
    parser.add_argument("--reweight-type",  type=str,   default='ones', choices=['ones','reciprocal','reciprocalinv'],     help="How to reweight the training")
    # --------------- sampling and evaluating ---------------
    parser.add_argument("--train-fid-sample",type=int, default=None,                                        help="number of samples used for evaluating FID")
    parser.add_argument("--sampling-batch",  type=int,   default=512,                                        help="mini batch size for gradient descent")
    parser.add_argument("--eval",           action="store_true",                                            help="evaluating mode. Wont save ckpt")
    parser.add_argument("--clip-x1",        action="store_true",                                            help="similar to DDPM, clip the estimiated data to [-1,1]")
    parser.add_argument("--debug",          action="store_true",                                            help="Using single GPU to evaluate. raise this flag for fast testing")
    parser.add_argument("--pred-x1",        action="store_true",                                            help="Using predict x1 as the sampling output")
    parser.add_argument("--gDDIM-r",        type=int,  default=2,                                           help="the checkpoint name from which we wish to sample")
    parser.add_argument("--diz-order",      type=int,  default=2,                                           help="the checkpoint name from which we wish to sample")
    parser.add_argument("--solver",         type=str,   default='em',              help="sampler")
    parser.add_argument("--diz",            type=str,   default='Euler', choices=['Euler','sigmoid'],       help="The discretization scheme")
    parser.add_argument("--sanity",         action="store_true",                                            help="quick sanity check for the proposed dyanmics") 
    # --------------- path and logging ---------------
    parser.add_argument("--log-dir",        type=Path,  default=".log",                                     help="path to log std outputs and writer data")
    parser.add_argument("--log-writer",     type=str,   default=None,        help="log writer: can be tensorbard, wandb, or None")
    parser.add_argument("--wandb-api-key",  type=str,   default=None,        help="unique API key of your W&B account; see https://wandb.ai/authorize")
    parser.add_argument("--wandb-user",     type=str,   default=None,        help="user name of your W&B account")
    

    default_config, model_configs = {
        'toy':          toy_config.get_toy_default_configs,
        'cifar10':      cifar10_config.get_cifar10_default_configs,
        'AFHQv2':      afhqv2_config.get_afhqv2_default_configs,
        'imagenet64':      imagenet64_config.get_imagenet64_default_configs,
    }.get(parser.parse_args().exp)()
    parser.set_defaults(**default_config)

    opt             = parser.parse_args()
    opt.model_config=model_configs

    # ========= auto setup =========
    opt.device      ='cuda' if opt.gpu is None else f'cuda:{opt.gpu}'
    opt.distributed = opt.n_gpu_per_node > 1

    if opt.solver=='sscs':
        assert opt.DE_type=='SDE'

    # log ngc meta data
    if "NGC_JOB_ID" in os.environ.keys():
        opt.ngc_job_id = os.environ["NGC_JOB_ID"]

    # ========= path handle =========
    os.makedirs(opt.log_dir, exist_ok=True)
    opt.ckpt_path       = RESULT_DIR / opt.name
    os.makedirs(opt.ckpt_path, exist_ok=True)

    if opt.train_fid_sample is  None:
        opt.train_fid_sample = opt.n_gpu_per_node*opt.microbatch

    if opt.ckpt is not None:
        ckpt_file = RESULT_DIR / opt.ckpt / "latest.pt"
        assert ckpt_file.exists()
        opt.load = ckpt_file
    else:
        opt.load = None


    return opt

def main(opt):
    log = Logger(opt.global_rank, opt.log_dir)
    log.info("=======================================================")
    log.info("        Accelerate Model")
    log.info("=======================================================")
    log.info("Command used:\n{}".format(" ".join(sys.argv)))
    log.info(f"Experiment ID: {opt.name}")

    # set seed: make sure each gpu has differnet seed!
    if opt.seed is not None:
        set_seed(opt.seed + opt.global_rank)

    # build dataset
    if opt.exp=='toy':
        train_loader   = spiral.spiral_data(opt)
    elif opt.exp=='cifar10':
        train_loader   = cifar10.cifar10_data(opt)
        if opt.cond: opt.cond_dim = 10
    elif opt.exp=='AFHQv2':
        train_loader   = AFHQv2.AFHQv2_data(opt)
    elif opt.exp=='imagenet64':
        train_loader   = imagenet64.imagenet64_data(opt)
        if opt.cond: opt.cond_dim = 1000

    run = runner.Runner(opt, log)
    if opt.eval:
        run.ema.copy_to()
        run.evaluation(opt,0,train_loader)
    else:
        run.train(opt, train_loader)
        log.info("Finish!")

if __name__ == '__main__':
    opt = create_training_options()
    if opt.debug:
        opt.distributed =False
        opt.global_rank = 0
        opt.local_rank  = 0
        opt.global_size = 1
        with torch.cuda.device(opt.gpu):
            main(opt)
    else:
        torch.multiprocessing.set_start_method('spawn')
        if opt.distributed:
            size        = opt.n_gpu_per_node
            processes   = []

            for rank in range(size):
                opt = copy.deepcopy(opt)
                opt.local_rank  = rank
                global_rank     = rank + opt.node_rank * opt.n_gpu_per_node
                global_size     = opt.num_proc_node * opt.n_gpu_per_node
                opt.global_rank = global_rank
                opt.global_size = global_size
                print('Node rank %d, local proc %d, global proc %d, global_size %d' % (opt.node_rank, rank, global_rank, global_size))

                p = Process(target=init_processes, args=(global_rank, global_size, main, opt))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            torch.cuda.set_device(0)
            opt.global_rank     = 0
            opt.local_rank      = 0
            opt.global_size     = 1
            init_processes(0, opt.n_gpu_per_node, main, opt)
