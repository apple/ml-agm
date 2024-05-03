#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
def save_toy_traj(opt, fn, traj):
    fn_pdf = os.path.join(opt.ckpt_path, fn+'.pdf')
    n_snapshot=2
    lims = [[-4, 4],[-6, 6]]

    total_steps = traj.shape[1]
    sample_steps= np.linspace(0, total_steps-1, n_snapshot).astype(int)
    traj_steps  = np.linspace(0, total_steps-1, 10).astype(int)
    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=10)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        fig, axss = plt.subplots(1, 2)
        fig.set_size_inches(20, 10)
        cmap =['Blues','Reds']
        colors= ['b','r']
        num_samp_lines = 10
        random_idx = np.random.choice(traj.shape[0], num_samp_lines, replace=False)
        means=traj[random_idx,...]
        for i in range(2):
            ax=axss[i]
            ax.grid(True)
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            _colors = np.linspace(0.5,1,len(sample_steps)) 
            for idx,step in enumerate(sample_steps):
                ax.scatter(traj[:,step,2*i],traj[:,step,2*i+1], s=10, c=_colors[idx].repeat(traj.shape[0]), alpha=0.6,vmin=0, vmax=1,cmap=cmap[i])
                ax.set_xlim(*lims[i])
                ax.set_ylim(*lims[i])

            for ii in range(num_samp_lines):
                ax.plot(means[ii,:,2*i],means[ii,:,2*i+1],color=colors[i],linewidth=4,alpha=0.5)
                ax.set_title('position' if i==0 else 'velocity',size=40)

    fig.suptitle('NFE = {}'.format(opt.nfe-1),size=40)
    fig.tight_layout()
    plt.savefig(fn_pdf)
    plt.clf()

def save_snapshot_traj(opt, fn, pred,gt):
    fn_pdf = os.path.join(opt.ckpt_path, fn+'_static.pdf')
    lims = [-4, 4]
    gt=gt.detach().cpu().numpy()
    pred=pred.detach().cpu().numpy()
    fig, axss = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    colors= ['b','r']
    ax=axss[0]
    ax.scatter(pred[:,0],pred[:,1],color='steelblue',alpha=0.3,s=5)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax=axss[1]
    ax.scatter(gt[:,0],gt[:,1],color='coral',alpha=0.3,s=5)
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    fig.suptitle('NFE = {}'.format(opt.nfe),size=40)
    fig.tight_layout()
    plt.savefig(fn_pdf)
    # np.save(fn_npy,traj)
    plt.clf()
    

def norm_data(x):
    bs=x.shape[0]
    _max=torch.max(torch.max(x,dim=-1)[0],dim=-1)[0][...,None,None]
    _min=torch.min(torch.min(x,dim=-1)[0],dim=-1)[0][...,None,None]
    x=(x-_min)/(_max-_min)
    return x


def plot_toy(opt,ms,it,pred_m1,x1):
    save_toy_traj(opt, 'itr_{}'.format(it), ms.detach().cpu().numpy())
    save_snapshot_traj(opt, 'itr_x{}'.format(it), pred_m1[:,0:2],x1)


def plot_scatter(x,ts,ax):
    '''
    x:bs,t,dim
    '''
    bs,interval,dim = x.shape
    for ii in range(interval):
        ax.scatter(ts[ii].repeat(bs),x[:,ii,:],s=2,color='b',alpha=0.1)

def plot_plt(x,ts,ax):
    '''
    x:bs,t,dim
    '''
    bs,interval,dim = x.shape
    
    for ii in range(bs):
        ax.plot(ts,x[ii,:,0],color='r',alpha=0.1)

def save_toy_npy_traj(opt, fn, traj, n_snapshot=None, direction='forward'):
    #form of traj: [bs, interval, x_dim=2]
    fn_pdf = os.path.join(opt.ckpt_path, fn+'.pdf')

    lims = [-5,5]

    if n_snapshot is None: # only store t=0
        plt.scatter(traj[:,0,0],traj[:,0,1], s=5)
        plt.xlim(*lims)
        plt.ylim(*lims)
    else:
        total_steps = traj.shape[1]
        sample_steps = np.linspace(0, total_steps-1, n_snapshot).astype(int)
        fig, axs = plt.subplots(1, n_snapshot)
        fig.set_size_inches(n_snapshot*6, 6)
        color = 'salmon' if direction=='forward' else 'royalblue'
        for ax, step in zip(axs, sample_steps):
            ax.scatter(traj[:,step,0],traj[:,step,1], s=1, color=color,alpha=0.2)
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_title('time = {:.2f}'.format(step/(total_steps-1)*opt.T))
        fig.tight_layout()

    plt.savefig(fn_pdf)
    plt.clf()