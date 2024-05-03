#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from prefetch_generator import BackgroundGenerator
import os
import warnings
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    warnings.warn("install your favorite tensorboard version")
import wandb
import termcolor
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import edm.distributed_util as dist_util
import abc
import numpy as np

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
def setup_loader(dataset, batch_size, num_workers=4):
    loader = DataLoaderX(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        persistent_workers=True,
        num_workers=num_workers,
        multiprocessing_context='spawn',
        drop_last=True,
        prefetch_factor=4,
    )
    # return loader
    while True:
        yield from loader




def save_ckpt(opt,log,it,net,ema,optimizer,sched,name):
    torch.save({
        "net": net.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "sched": sched.state_dict() if sched is not None else sched,
    }, opt.ckpt_path / name)
    log.info(f"Saved latest({it=}) checkpoint to {opt.ckpt_path=}!")

class BaseWriter(object):
    def __init__(self, opt):
        self.rank = opt.global_rank
    def add_scalar(self, step, key, val):
        pass # do nothing
    def add_image(self, step, key, image):
        pass # do nothing
    def add_bar(self, step, key, image):
        pass
    def close(self): pass
    
class WandBWriter(BaseWriter):
    def __init__(self, opt):
        super(WandBWriter,self).__init__(opt)
        if self.rank == 0:
            assert wandb.login(key=opt.wandb_api_key)
            wandb.init(dir=str(opt.log_dir), project="VM", entity=opt.wandb_user, name=opt.name, config=vars(opt))

    def add_scalar(self, step, key, val):
        if self.rank == 0: wandb.log({key: val}, step=step)

    def add_image(self, step, key, image):
        if self.rank == 0:
            # adopt from torchvision.utils.save_image
            image = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            wandb.log({key: wandb.Image(image)}, step=step)

    def add_bar(self,step,key,data):
        if self.rank == 0:
            fig, ax = plt.subplots()
            # plt.ylim([0,5])
            ts,loss=data
            ax.bar(ts, loss)
            wandb.log({"plot": wandb.Image(fig)},step=step)


class TensorBoardWriter(BaseWriter):
    def __init__(self, opt):
        super(TensorBoardWriter,self).__init__(opt)
        if self.rank == 0:
            run_dir = str(opt.log_dir / opt.name)
            os.makedirs(run_dir, exist_ok=True)
            self.writer=SummaryWriter(log_dir=run_dir, flush_secs=20)

    def add_scalar(self, global_step, key, val):
        if self.rank == 0: self.writer.add_scalar(key, val, global_step=global_step)

    def add_image(self, global_step, key, image):
        if self.rank == 0:
            image = image.mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)
            self.writer.add_image(key, image, global_step=global_step)

    def close(self):
        if self.rank == 0: self.writer.close()

def build_log_writer(opt):
    if opt.log_writer == 'wandb': return WandBWriter(opt)
    elif opt.log_writer == 'tensorboard': return TensorBoardWriter(opt)
    else: return BaseWriter(opt) # do nothing

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def space_indices(num_steps, count):
    assert count <= num_steps

    if count <= 1:
        frac_stride = 1
    else:
        frac_stride = (num_steps - 1) / (count - 1)

    cur_idx = 0.0
    taken_steps = []
    for _ in range(count):
        taken_steps.append(round(cur_idx))
        cur_idx += frac_stride

    return taken_steps

def unsqueeze_xdim(z, xdim):
    bc_dim = (...,) + (None,) * len(xdim)
    return z[bc_dim]

def merge(opt,x,v):
    dim=-1 if opt.exp=='toy' else -3
    return torch.cat([x,v], dim=-1)

def flatten_dim01(x):
    # (dim0, dim1, *dim2) --> (dim0x1, *dim2)
    return x.reshape(-1, *x.shape[2:])

# convert to colored strings
def red(content): return termcolor.colored(str(content),"red",attrs=["bold"])
def green(content): return termcolor.colored(str(content),"green",attrs=["bold"])
def blue(content): return termcolor.colored(str(content),"blue",attrs=["bold"])
def cyan(content): return termcolor.colored(str(content),"cyan",attrs=["bold"])
def yellow(content): return termcolor.colored(str(content),"yellow",attrs=["bold"])
def magenta(content): return termcolor.colored(str(content),"magenta",attrs=["bold"])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reshape_as(x,y):
    len_y = len(y.shape)-1
    return x.reshape(-1,*([1,]*len_y))

def all_cat_cpu(opt, log, t):
    if not opt.distributed: return t.detach().cpu()
    log_flag = log if opt.local_rank == 0 else None
    gathered_t = dist_util.all_gather(t.to(opt.device), log=log_flag)
    return torch.cat(gathered_t).detach().cpu()

def cast_shape(x,dims):
    return x.reshape(-1,*([1,]*len(dims)))

def uniform_ts(t0,T,n,device):
    _ts     = uniform(n,t0,T,device)
    return _ts
    
def debug_ts(t0,T,n,device):
    return torch.linspace(t0, T, n, device=device)

def heuristic2_ts(t0,T,n,device):
    _ts = torch.randn(n,device=device)*0.1+T
    _ts = _ts.abs()
    invalid_item=torch.logical_or(_ts>T, _ts<t0)
    _ts[invalid_item] = uniform(invalid_item.sum(),t0,T,device)
    return _ts

def heuristic3_ts(t0,T,n,device):
    _ts = torch.randn(n,device=device)*0.5+0.3
    _ts = _ts.abs()
    invalid_item=torch.logical_or(_ts>T, _ts<t0)
    _ts[invalid_item] = uniform(invalid_item.sum(),t0,T,device)
    return _ts

def heuristic_ts(t0,T,n,device):
    _ts = torch.randn(n,device=device)*0.2+0.5
    _ts = _ts.abs()
    invalid_item=torch.logical_or(_ts>T, _ts<t0)
    _ts[invalid_item] = uniform(invalid_item.sum(),t0,T,device)
    return _ts

def uniform(bs,r1,r2,device):
    return (r1 - r2) * torch.rand(bs,device=device) + r2
