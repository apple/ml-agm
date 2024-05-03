#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import gc
import os
import gdown
import copy
import argparse
import random
from pathlib import Path
from easydict import EasyDict as edict
import PIL.Image
import numpy as np
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from networks.get_network import get_nn
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
from AM.dynamics import TVgMPC
from AM.samplers import SamplerWrapper
from edm import dnnlib
from edm.logger import Logger
import edm.distributed_util as dist_util
from AM import runner,util
from AM.util import all_cat_cpu
import colored_traceback.always
import glob
import pickle
import string
import random
from edm.fid import calculate_inception_stats,calculate_fid_from_inception_stats

def id_generator(size=4, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
    
RESULT_DIR  = Path("results")
FID_DIR     = Path("FID_EVAL")


def build_ckpt_option(opt, log, ckpt_path,ckpt_file):
    ckpt_path = Path(ckpt_path)
    opt_pkl_path = ckpt_path / "options.pkl"
    assert opt_pkl_path.exists()
    with open(opt_pkl_path, "rb") as f:
        ckpt_opt = pickle.load(f)
    log.info(f"Loaded options from {opt_pkl_path=}!")

    overwrite_keys = ["use_fp16", "device","solver"]
    for k in overwrite_keys:
        assert hasattr(opt, k)
        setattr(ckpt_opt, k, getattr(opt, k))

    ckpt_opt.load = ckpt_file
    if hasattr(ckpt_opt,'ode') and ckpt_opt.ode==True:
        ckpt_opt.DE_type='ODE'
    elif hasattr(ckpt_opt,'probablistic_ode') and ckpt_opt.probablistic_ode==True:
        ckpt_opt.DE_type='probODE'

    if not hasattr(ckpt_opt,'cond'):
        ckpt_opt.cond       = False
        sampling_opt.cond   = False
    else:
        sampling_opt.cond   = ckpt_opt.cond
        if sampling_opt.cond: sampling_opt.cond_dim = ckpt_opt.cond_dim
    
    if sampling_opt.batch_size is not None: ckpt_opt.sampling_batch = sampling_opt.batch_size

    return ckpt_opt


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



def dist_handle(opt,log,x):
    #Distributed training handle
    return all_cat_cpu(opt, log, x.contiguous()) if opt.distributed else x.contiguous().cpu()

def loop_saving_png(opt,sampler,ema,net,log,image_dir,return_last=True,num_loop=1,normalize_data=False):
    os.makedirs(image_dir, exist_ok=True)
    num_loop = tqdm(range(num_loop), position=1, desc=util.green("Sampling loop"), colour='green')
    # with run.ema.average_parameters():
    img_idx=0
    try:
        batchsize = opt.batch_size
    except:
        batchsize = opt.sampling_batch

    for loop in (num_loop):
        m0      =    sampler.dyn.get_m0(batchsize).to(opt.device)
        if opt.cond:
            class_cond  = F.one_hot(torch.randint(0,opt.cond_dim,(m0.shape[0],)),num_classes=opt.cond_dim).to(torch.float32).to(opt.device)
        else:
            class_cond  = None
            
        ms,pred_m1,est_x1s, snapts\
                = sampler.solve(ema,net,m0,cond=class_cond)

        out_x1  = est_x1s[:,-1,...] if opt.pred_x1  else pred_m1
        out_x1  = dist_handle(opt,log,out_x1[:,0:3,...])
        ms      = dist_handle(opt,log,ms)
        snapts  = dist_handle(opt,log,snapts)

        if normalize_data:
            for ii in range(out_x1.shape[0]):
                img=out_x1[ii]
                image_path = os.path.join(image_dir, 'num_{}.png'.format(img_idx))
                tu.save_image((img+1)/2, image_dir+'/num_{}.png'.format(img_idx),scale_each=True)
                img_idx+=1
        else:
            out_x1 = (out_x1 * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for ii in range(out_x1.shape[0]):
                image_path = os.path.join(image_dir, 'num_{}.png'.format(img_idx))
                PIL.Image.fromarray(out_x1[ii], 'RGB').save(image_path)
                img_idx+=1

        if return_last:
            _pred_m1=pred_m1[0:100]
            _ms     = ms[0:100]
            _est_x1s=est_x1s[0:100]
            _snapts = snapts[0:100]
        del ms, pred_m1, out_x1,snapts
        torch.cuda.empty_cache()
        gc.collect()
    if return_last: 
        return _ms, _pred_m1, _est_x1s,_snapts

       
@torch.no_grad()
def main(sampling_opt):
    log = Logger(sampling_opt.global_rank, ".log")
    log.info(f"loading path from {sampling_opt.ckpt_path}!")
    log.info(f"loading file from {sampling_opt.ckpt_file}!")
    # if sampling_opt.eval_fid:
    log.info(f"fid evalute folder is {sampling_opt.fid_dir}!")
    # get (default) ckpt option
    ckpt_opt = build_ckpt_option(sampling_opt, log, sampling_opt.ckpt_path,sampling_opt.ckpt_file)

    '''
    set up dynamics
    '''
    dyn_kargs   = {
        "p":ckpt_opt.p, #diffusion coeffcient
        'k':ckpt_opt.k, # covariance of prior
        'varx':ckpt_opt.varx,
        'varv':ckpt_opt.varv,
        'x_dim':ckpt_opt.data_dim,
        'device':sampling_opt.device, #Using sampling device
        'DE_type':ckpt_opt.DE_type
    } 
    dynamics = TVgMPC(**dyn_kargs)

    '''
    set up dynamics solver
    '''
    
    solver_kargs   = {
        "solver_name":sampling_opt.solver, #updated solver
        'diz':sampling_opt.diz, #updated diz
        't0':ckpt_opt.t0, # original t0
        'T':sampling_opt.T, # updated T
        'interval':sampling_opt.nfe, #updated NFE
        'dyn': dynamics, #updated dynamics
        'device':sampling_opt.device, #Using sampling device
        'snap': 10,
        'local_rank':sampling_opt.local_rank,
        'diz_order': sampling_opt.diz_order,
        'gDDIM_r':sampling_opt.gDDIM_r,
        'cond_opt': sampling_opt.cond_opt
    }
    sampler    = SamplerWrapper(**solver_kargs)

    '''
    set up networks
    '''
    net = get_nn(ckpt_opt,dynamics)
    ema = ExponentialMovingAverage(net.parameters(),decay=0.999)
    checkpoint = torch.load(sampling_opt.ckpt_file, map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    ema.load_state_dict(checkpoint['ema'])
    net.to(sampling_opt.device)
    ema.to(sampling_opt.device)
    net.eval()


    img_dir = os.path.join(sampling_opt.ckpt_path/ sampling_opt.img_save_name)
    os.makedirs(img_dir, exist_ok=True)

    log.info('Using Sampler{}'.format(sampling_opt.solver))
    num_loop=int((sampling_opt.num_sample-1)/sampling_opt.n_gpu_per_node/sampling_opt.batch_size)+1
    log.info('num of loop {}, batch {}, number of gpu{}, sampling number {}'.format(num_loop, sampling_opt.batch_size, sampling_opt.n_gpu_per_node,sampling_opt.num_sample))

    ms,pred_m1, est_x1s, snapts = loop_saving_png(\
                                    sampling_opt,\
                                    sampler,\
                                    ema,\
                                    net,\
                                    log,\
                                    sampling_opt.fid_dir,\
                                    return_last=True,\
                                    num_loop=num_loop,\
                                    normalize_data=sampling_opt.normalize)
    
    if sampling_opt.save_img:
        pred_x1     = all_cat_cpu(sampling_opt, log, pred_m1[:,0:3,...].contiguous())
        pred_v1     = all_cat_cpu(sampling_opt, log, pred_m1[:,3:,...].contiguous())
        nrow=10
        pred_x1=pred_m1[:,0:3,...]
        pred_v1=pred_m1[:,3:,...]
        gt_ts       = snapts
        _pos_traj   = ms[0:5,:,0:3,...]
        pos_traj    = _pos_traj.reshape(-1,*ckpt_opt.data_dim)
        vel_traj    = ms[0:5,:,3:,...]
        est_x1s     = est_x1s[0:5,:,:,...]
        vel_traj    = vel_traj.reshape(-1,*ckpt_opt.data_dim)
        est_x1s     = est_x1s.reshape(-1,*ckpt_opt.data_dim)
        num_samp    = 200
        if sampling_opt.cond_opt:
            tu.save_image((((sampling_opt.cond_opt.stroke)+1)/2), img_dir+'/cond_data.png', nrow = 1)    
        tu.save_image((((pred_x1[0:num_samp])+1)/2), img_dir+'/outputx1.png', nrow = 10)
        tu.save_image((((pred_v1[0:num_samp])+1)/2), img_dir+'/v.png', nrow = 11)
        tu.save_image((((est_x1s[0:num_samp])+1)/2), img_dir+'/estx1.png', nrow = 11)
        tu.save_image((((pos_traj)+1)/2), img_dir+'/pos_traj.png', nrow = 11)
        tu.save_image((((vel_traj)+1)/2), img_dir+'/vel_traj.png', nrow = 11)
        if sampling_opt.save_npy:
            np.save(img_dir+'/estx1.npy',(est_x1s.detach().cpu().numpy()+1)/2)
            np.save(img_dir+'/pos_traj.npy',(pos_traj.detach().cpu().numpy()+1)/2)
            np.save(img_dir+'/vel_traj.npy',(vel_traj.detach().cpu().numpy()+1)/2)

    if sampling_opt.eval_fid:
        mu,sigma=calculate_inception_stats(image_path=sampling_opt.fid_dir,num_expected=sampling_opt.num_sample,seed=42)
        ref_file_name ={'cifar10':'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz',
                        'AFHQv2': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz',
                        'imagenet64': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz'
                        }.get(ckpt_opt.exp)
        with dnnlib.util.open_url(ref_file_name) as f:
            ref = dict(np.load(f))
        fid = calculate_fid_from_inception_stats(mu,sigma,ref['mu'], ref['sigma'])
        log.info(f"========== FID is: {fid} ==========")
        torch.cuda.empty_cache()
        dist.barrier()

        log.info(f"Sampling complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,  default=999)
    parser.add_argument("--n-gpu-per-node",     type=int,  default=1,           help="number of gpu on each node")
    parser.add_argument("--master-address",     type=str,  default='localhost', help="address for master")
    parser.add_argument("--node-rank",          type=int,  default=0,           help="the index of node")
    parser.add_argument("--num-proc-node",      type=int,  default=1,           help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size",         type=int,  default=32)
    parser.add_argument("--gpu",                type=int,  default=0)
    parser.add_argument("--dataset-dir",        type=Path, default="/dataset",  help="path to LMDB dataset")
    parser.add_argument("--img-save-name",      type=str,  default="eval",        help="e.g., '0_4' means the first 25% of the dataset")
    parser.add_argument("--fid-save-name",      type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")
    parser.add_argument("--DE-type",            type=str,  default=None,        help="e.g., '0_4' means the first 25% of the dataset")

    # sample
    parser.add_argument("--batch-size",         type=int,  default=32)
    parser.add_argument("--diz-order",          type=float,  default=2)
    parser.add_argument("--diz",                type=str,  default='rev-quad',        help="diz type")
    parser.add_argument("--num-sample",         type=int,  default=50000)
    parser.add_argument("--solver",             type=str,  default='em',        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--ckpt",               type=str,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--T",                  type=float,  default=None,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--gDDIM-r",            type=int,  default=2,        help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe",                type=int,  default=None,        help="sampling steps")
    parser.add_argument("--port",               type=str,  default='6022',        help="sampling steps")
    parser.add_argument("--last",               type=int,  default=-1,        help="sampling steps")
    parser.add_argument("--start-point",        type=int,  default=0,        help="sampling steps")
    parser.add_argument("--probablistic-ode",   type=bool,  default=False,        help="sampling steps")
    parser.add_argument("--save-img",           action="store_true",            help="clamp predicted image to [-1,1] at each")

    parser.add_argument("--stroke-path",        type=str,    default=None,        help="Whether using stroke guide generation. input your guidence picture")
    parser.add_argument("--stroke-type",        type=str,    default=None,        help="how to do the stroke")
    parser.add_argument("--impainting",         action="store_true",            help="dooing impainting task")

    parser.add_argument("--eval-fid",           action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--clip-x1",            action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--normalize",          action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--debug",              action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--clip-denoise",       action="store_true",            help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16",           action="store_true",            help="use fp16 network weight for faster sampling")
    parser.add_argument("--denoise",            action="store_true",             help="add noise to conditional network")
    parser.add_argument("--pred-x1",            action="store_true",             help="add noise to conditional network")
    parser.add_argument("--save-npy",           action="store_true",             help="add noise to conditional network")
    
    arg = parser.parse_args()

    sampling_opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    sampling_opt.update(vars(arg))

    sampling_opt.cond_opt = None
    if sampling_opt.stroke_path:
        assert sampling_opt.stroke_type is not None
        stroke = torch.Tensor((plt.imread(sampling_opt.stroke_path)[None,:,:,0:3]).transpose(0,3,1,2)*2-1).to(sampling_opt.device)
        cond_opt= edict(
            stroke          = stroke,
            stroke_type     = sampling_opt.stroke_type,
            impainting      = sampling_opt.impainting
        )
        sampling_opt.cond_opt = cond_opt


    if 'Remote' == sampling_opt.ckpt[0:6]:
        try:
            folder_id={
                "Remote_Cifar10_ODE":'1G6gRF269F6di4DZHwlj72FaE-I6lgb92?usp=sharing',
                "Remote_AFHQv2_ODE":'1VkVfshRk7Ca5FVFQzJgDr0lA49LrJ8X6?usp=sharing',
                "Remote_uncondImageNet64_ODE":'1Ubd6Q-wtGaCzd2blmxIefrpnBzZd1M2P?usp=sharing',
            }.get(sampling_opt.ckpt)
            print(util.magenta('=======Downloading checkpoint from remote ======='))
            gdown.download_folder(  id          =folder_id,\
                                    output      ="results/LocalVersion_{}".format(sampling_opt.ckpt),\
                                    quiet       =True,\
                                    use_cookies =False)
            folder_name="LocalVersion_{}".format(sampling_opt.ckpt)
            sampling_opt.ckpt = "LocalVersion_{}/latest.pt".format(sampling_opt.ckpt)
            print(util.magenta('=======downloaded Checkpoint ======='))
        except:
            print(util.magenta('=======And error occured in downloading remote ckpt, try local... ======='))
            sampling_opt.ckpt = "LocalVersion_{}/latest.pt".format(sampling_opt.ckpt)
            latest_ckpt = sampling_opt.ckpt.split('/')[-1]
            folder_name=''
            for idx,item in enumerate(sampling_opt.ckpt.split('/')):
                if idx !=len(sampling_opt.ckpt.split('/'))-1:
                    folder_name=os.path.join(folder_name,item)                
    else:
        latest_ckpt = sampling_opt.ckpt.split('/')[-1]
        folder_name=''
        for idx,item in enumerate(sampling_opt.ckpt.split('/')):
            if idx !=len(sampling_opt.ckpt.split('/'))-1:
                folder_name=os.path.join(folder_name,item)

    ckpt_dir            = RESULT_DIR/folder_name
    sampling_opt.ckpt_file       = RESULT_DIR / sampling_opt.ckpt
    sampling_opt.ckpt_path       = ckpt_dir

    fid_folder_id   = id_generator() if sampling_opt.fid_save_name is None else sampling_opt.fid_save_name
    fid_dir = os.path.join(FID_DIR,fid_folder_id)
    os.makedirs(fid_dir, exist_ok=True)
    sampling_opt.fid_dir=fid_dir

    set_seed(sampling_opt.seed)

    if sampling_opt.debug:
        sampling_opt.distributed=False
        sampling_opt.global_rank = 0
        sampling_opt.local_rank = 0
        sampling_opt.global_size = 1
        with torch.cuda.device('cuda:{}'.format(sampling_opt.gpu)):
            main(sampling_opt)
    else:            
        torch.multiprocessing.set_start_method('spawn',force=True)
        if sampling_opt.distributed:
            size = sampling_opt.n_gpu_per_node

            processes = []
            for rank in range(size):
                sampling_opt = copy.deepcopy(sampling_opt)
                sampling_opt.local_rank = rank
                global_rank = rank + sampling_opt.node_rank * sampling_opt.n_gpu_per_node
                global_size = sampling_opt.num_proc_node * sampling_opt.n_gpu_per_node
                sampling_opt.global_rank = global_rank
                sampling_opt.global_size = global_size
                print('Node rank %d, local proc %d, global proc %d, global_size %d' % (sampling_opt.node_rank, rank, global_rank, global_size))
                p = Process(target=dist_util.init_processes, args=(global_rank, global_size, main, sampling_opt))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            torch.cuda.set_device(0)
            sampling_opt.global_rank = 0
            sampling_opt.local_rank = 0
            sampling_opt.global_size = 1
            dist_util.init_processes(0, sampling_opt.n_gpu_per_node, main, sampling_opt)

