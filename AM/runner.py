#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import os
import numpy as np
import pickle
import torch
import pytorch_warmup as warmup
import torch.nn.functional as F
from torch.optim import AdamW, lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP
from edm import dnnlib
from torch_ema import ExponentialMovingAverage
import torchvision.utils as tu
# from .get_network import get_nn
from networks.get_network import get_nn
from . import util
import plot_util
from .diffusion import Diffusion
from plot_util import norm_data, plot_plt,plot_scatter
from .util import all_cat_cpu
from edm.fid import calculate_fid_from_inception_stats, calculate_inception_stats
from sampling import loop_saving_png


def build_optimizer_sched(opt, net, log):
    optim_dict  = {"lr": opt.lr, 'weight_decay': opt.l2_norm}
    optimizer   = AdamW(net.parameters(), **optim_dict)
    log.info(f"[Opt] Built AdamW optimizer {optim_dict=}!")

    sched = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.num_itr)
    warmup_scheduler = warmup.UntunedLinearWarmup(optimizer)
    log.info(f"[Opt] Built lr _step scheduler Cosine")


    if opt.load:
        checkpoint = torch.load(opt.load, map_location="cpu")
        if "optimizer" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint["optimizer"])
            log.info(f"[Opt] Loaded optimizer ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no optimizer!")
        if sched is not None and "sched" in checkpoint.keys() and checkpoint["sched"] is not None:
            sched.load_state_dict(checkpoint["sched"])
            log.info(f"[Opt] Loaded lr sched ckpt {opt.load}!")
        else:
            log.warning(f"[Opt] Ckpt {opt.load} has no lr sched!")
        for g in optimizer.param_groups:
            g['lr'] = opt.lr

    return optimizer, sched,warmup_scheduler



class Runner(object):
    def __init__(self, opt, log, save_opt=True):
        super(Runner,self).__init__()

        # ===========Save opt. ===========
        if save_opt:
            opt_pkl_path    = opt.ckpt_path / "options.pkl"
            with open(opt_pkl_path, "wb") as f:
                pickle.dump(opt, f)
            log.info("Saved options pickle to {}!".format(opt_pkl_path))

        self.diffusion = Diffusion(opt, opt.device)
        if opt.exp!='toy':
            ref_file_name ={'cifar10':'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz',
                            'AFHQv2': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz',
                            'imagenet64': 'https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/imagenet-64x64.npz',
                            }.get(opt.exp)

            with dnnlib.util.open_url(ref_file_name) as f:
                self.ref = dict(np.load(f))

        self.net        = get_nn(opt,self.diffusion.dyn)
        log.info('network size [{}]'.format(util.count_parameters(self.net)))
        self.ema        = ExponentialMovingAverage(self.net.parameters(), decay=opt.ema)
        self.opt        = opt
        self.reweight   = self.diffusion.reweights 
        self.ts_sampler = {'uniform':util.uniform_ts,
                            'debug':util.debug_ts,
                            }.get(opt.t_samp)
        if opt.load:
            checkpoint = torch.load(opt.load, map_location="cpu")
            self.net.load_state_dict(checkpoint['net'])
            log.info(f"[Net] Loaded network ckpt: {opt.load}!")
            self.ema.load_state_dict(checkpoint["ema"])
            log.info(f"[Ema] Loaded ema ckpt: {opt.load}!")

        self.net.to(opt.device)
        self.ema.to(opt.device)
        self.log        = log
        self.best_res   = np.inf
        
        
    def train(self, opt, train_loader):
        self.writer = util.build_log_writer(opt)
        log = self.log

        if opt.distributed:
            net = DDP(self.net, device_ids=[opt.device])
        else:
            net=self.net

        ema = self.ema
        optimizer, sched, warmup    = build_optimizer_sched(opt, net, log)
        ts_sampler, reweight        = self.ts_sampler, self.reweight
        t0,T,device                 = opt.t0, opt.T, opt.device

        net.train()

        for it in range(opt.num_itr):
            optimizer.zero_grad(set_to_none=True)
            # ===== sample boundary pair =====
            x1,class_cond\
                    = train_loader.sample()
            # ===== compute loss =====
            _ts     = ts_sampler(t0,T,x1.shape[0],device) 
            label,mt= self.diffusion.mt_sample(x1,ts=_ts)
            lambdat = reweight(_ts)[:,None]
            lambdat = lambdat.reshape(-1,*([1,]*(len(x1.shape)-1)))
            pred    = net(mt, _ts,cond=class_cond)
            label   = label.reshape_as(pred)
            _pred   = pred.detach().cpu() #for rendering loss over time
            _label  = label.detach().cpu() #for rendering loss over time
            loss = F.mse_loss(lambdat*pred, lambdat*label)
            loss.backward()

            if opt.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(net.parameters(), opt.clip_grad)
            optimizer.step()
            ema.update()

            if sched is not None: 
                with warmup.dampening():
                    sched.step()
  
            # # -------- logging --------
            log.info("train_it {}/{} | lr:{} | loss:{}".format(
                1+it,
                opt.num_itr,
                "{:.2e}".format(optimizer.param_groups[0]['lr']),
                "{:+.4f}".format(loss.item()),
            ))
            if it % 10 == 0:
                self.writer.add_scalar(it, 'loss', loss.detach())

            #============monitoring the loss distribution over time=======
            if it% 1000 ==0 and self.writer is not None: 
                total_idxs          = torch.arange(0,opt.nfe).float()
                total_vals          = torch.zeros_like(total_idxs).float()
                _lambdat            = lambdat.detach().cpu()
                if self.opt.exp=='toy':
                    _loss               = (((_pred*_lambdat-_label*_lambdat)**2).sum(-1)).float()
                else:
                    _loss               = (((_pred*_lambdat-_label*_lambdat)**2).sum(-1).sum(-1).sum(-1)).float()
                total_vals[(_ts*opt.nfe).long()]  = _loss/(_pred.numel()/_pred.shape[0])
                

                self.writer.add_bar(it,'ts_loss',[total_idxs,total_vals])
            #============monitoring the loss distribution over time=======

            if it == 1000 or it % 10000 == 0:
            # if it >=9999 and it % 5000 == 0:
                net.eval()
                results=self.evaluation(opt, it,train_loader)
                net.train()

                if results<=self.best_res:
                    self.best_res=results
                    name = "latest.pt".format(it)
                    util.save_ckpt(opt,log,it,self.net,ema,optimizer,sched,name) #Using Self.net to handle DDP

            if it%10000==0:
                if opt.global_rank == 0:
                    name = "latest_it{}.pt".format(it)
                    util.save_ckpt(opt,log,it,self.net,ema,optimizer,sched,name)

                if opt.distributed:
                    torch.distributed.barrier()

        self.writer.close()


    @torch.no_grad()
    def evaluation(self, opt, it, loader):
        log = self.log
        log.info(f"========== Evaluation started: iter={it} ==========")

        def log_image(tag, img, nrow=10):
            self.writer.add_image(it, tag, tu.make_grid((img+1)/2, nrow=nrow,scale_each=True)) # [1,1] -> [0,1]

        x1,class_cond = loader.sample()
        if opt.exp=='toy':
            sampler = self.diffusion.sampler
            m0      =    sampler.dyn.get_m0(opt.sampling_batch).to(opt.device)
            ms, pred_m1,est_x1s,snap_ts = sampler.solve(self.ema,self.net,m0,cond=None)
            plot_util.plot_toy(opt,ms,it,pred_m1,x1)
            pos_traj     = ms[:,:-1,0:2,...]
            vel_traj    = ms[:,:-1,2:,...]
            est_x1s     = est_x1s.detach().cpu().numpy()
            plot_util.save_toy_npy_traj(opt,'itr_{}_est_x1s'.format(it),est_x1s,n_snapshot=10)
        else:
            image_dir   = os.path.join(opt.ckpt_path , 'fid_train_folder')
            num_loop=int((opt.train_fid_sample-1)/opt.num_proc_node/opt.n_gpu_per_node/opt.sampling_batch)+1
            log.info('num of loop {}, batch {}, number of gpu{}, sampling number {}'.format(num_loop, opt.sampling_batch, opt.n_gpu_per_node,opt.train_fid_sample))

            ms, pred_m1, est_x1s,snap_ts = loop_saving_png( opt,\
                                                            self.diffusion.sampler,\
                                                            self.ema,\
                                                            self.net,\
                                                            log,\
                                                            image_dir,\
                                                            num_loop=num_loop,\
                                                            return_last=True)

            mu,sigma=calculate_inception_stats( image_path=image_dir,\
                                                num_expected=opt.train_fid_sample,\
                                                seed=42,\
                                                max_batch_size=128)
            
            fid = calculate_fid_from_inception_stats(mu,sigma,self.ref['mu'], self.ref['sigma'])
            log.info(f"========== FID is: iter={fid} ==========")
            log.info(f"========== FID folder is at : {image_dir} ==========")
            self.writer.add_scalar(it, 'fid', fid)
            ########Visualizing resulting data########
            num_samp=40
            pred_x1     =   pred_m1[:,0:3,...]
            pred_v1     =   pred_m1[:,3:,...]
            est_x1s     =   est_x1s
            ms          =   ms
            gt_x1       =   x1
            gt_ts       =   snap_ts
            
            if opt.log_writer is not None:
                _pos_traj   = ms[0:5,:,0:3,...]
                pos_traj    = _pos_traj.reshape(-1,*opt.data_dim)
                vel_traj    = ms[0:5,:,3:,...]
                est_x1s     = est_x1s[0:5,:,:,...]
                vel_traj    = vel_traj.reshape(-1,*opt.data_dim)
                est_x1s     = est_x1s.reshape(-1,*opt.data_dim)
                log_image("image/position",   (pred_x1[0:num_samp]))
                log_image("image/velocity",   (pred_v1[0:num_samp]))
                log_image("image/gt",   (gt_x1[0:num_samp]))
                log_image("image/position_traj",pos_traj,nrow=11)
                log_image("image/velocity_traj",vel_traj,nrow=11)
                log_image("image/est_x1s",est_x1s,nrow=11)
            else:
                fn_pdf = os.path.join(opt.ckpt_path, 'itr_{}_x.png'.format(it))
                tu.save_image(norm_data((pred_x1+1)/2), fn_pdf, nrow = 6)
                fn_pdf = os.path.join(opt.ckpt_path, 'itr_{}_v.png'.format(it))
                tu.save_image(norm_data((pred_v1+1)/2), fn_pdf, nrow = 6)
                if it==0:
                    fn_pdf = os.path.join(opt.ckpt_path, 'itr_{}_ground_truth.png'.format(it))
                    tu.save_image(norm_data((gt_x1+1)/2), fn_pdf, nrow = 6)
        ########Visualizing resulting data########
        log.info(f"========== Evaluation finished: iter={it} ==========")
        torch.cuda.empty_cache()
        results = self.best_res if opt.exp=='toy' else fid
        return results
    