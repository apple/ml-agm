#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from networks.network import *
from networks.edm.ncsnpp import SongUNet,DhariwalUNet
import torch
# from .util import Ltvv, Ltxx,Ltxv,reshape_as
def get_nn(opt,dyn):
    net={
        'toy': ResNet,
        'cifar10': SongUNet,
        'AFHQv2': SongUNet,
        'imagenet64':DhariwalUNet
    }.get(opt.exp)
    return network_wrapper(opt,net(opt),dyn)

class network_wrapper(torch.nn.Module):
    # note: scale_by_g matters only for pre-trained model
    def __init__(self, opt, net,dyn):
        super(network_wrapper,self).__init__()
        self.opt = opt
        self.net = net
        self.dim = self.opt.data_dim
        self.varx= opt.varx
        self.varv= opt.varv
        self.dyn= dyn
        self.p   = opt.p

    def get_precond(self,m,t):
        precond = (1-t)
        precond =precond.reshape(-1,*([1,]*(len(m.shape)-1)))
        return precond


    def forward(self, m, t,cond=None):
        t = t.squeeze()
        if t.dim()==0: t = t.repeat(m.shape[0])
        assert t.dim()==1 and t.shape[0] == m.shape[0]
        precond = 1
        if self.opt.precond: precond =self.get_precond(m,t)
        out = precond*self.net(m, 1-t,class_labels=cond) #Flip the time because we are generating image at t=1. It will influence the time cond (log t).
        return out
