#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from .samplers import SamplerWrapper
from .dynamics import TVgMPC
import torch.nn.functional as F

class Diffusion():    
    def __init__(self, opt, device):
        self.opt        = opt
        self.device     = device
        self.varx       = opt.varx
        self.varv       = opt.varv
        dyn_kargs   = {
            "p":opt.p, #diffusion coeffcient
            'k':opt.k, # covariance of prior
            'varx':opt.varx,
            'varv':opt.varv,
            'x_dim':opt.data_dim,
            'device':opt.device, #Using sampling device
            'DE_type':opt.DE_type
        } 
        dynamics = TVgMPC(**dyn_kargs)

        '''
        set up dynamics solver
        '''
        solver_kargs   = {
            "solver_name":opt.solver, #updated solver
            'diz':opt.diz, #updated diz
            't0':opt.t0, # original t0
            'T':opt.T, # updated T
            'interval':opt.nfe, #updated NFE
            'dyn': dynamics, #updated dynamics
            'device':opt.device, #Using sampling device
            'snap': 10,
            'local_rank':opt.local_rank,
            'diz_order': opt.diz_order,
            'gDDIM_r':opt.gDDIM_r,
            'cond_opt':None
        }

        self.sampler    = SamplerWrapper(**solver_kargs)
        self.dyn        = TVgMPC(opt.p,opt.k,opt.varx,opt.varv,opt.data_dim,opt.device,opt.DE_type)

        if self.opt.local_rank ==0:
            print('----------using sampling method as {}'.format(opt.solver))

    def reweights(self, t):
        reweight_type = self.opt.reweight_type
        dyn    =  self.dyn
        x_dim   = self.opt.data_dim
        if reweight_type =='ones':
            return torch.ones_like(t)
        elif reweight_type=='reciprocal':
            weight = 1/(1-t)
            return torch.sqrt(weight)
        elif reweight_type=='reciprocalinv':
            weight = 1/(t)
            return torch.sqrt(weight)
        else:
            raise RuntimeError
    


    def mt_sample(self,x1,ts): 
        """ return xs.shape == [batch_x, batch_t, *x_dim]  """
        opt         = self.opt
        dyn         = self.dyn
        x_dim       = self.opt.data_dim
        joint_dim   = self.opt.joint_dim
        t           = ts.reshape(-1,*([1,]*len(x_dim)))
        analytic_xs, analytic_vs, analytic_fv =dyn.get_xt_vt_fv(t,x1,opt.DE_type,opt.device)
        
        analytic_ms = torch.cat([analytic_xs,analytic_vs],dim=1)
        label       = analytic_fv
        inputs      = analytic_ms
        return label, inputs

