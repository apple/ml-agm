#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import abc
import torch
import numpy as np
from .util import cast_shape
class BaseDynamics(metaclass=abc.ABCMeta):
    def __init__(self, p,k,varx,varv,x_dim,device):
        self.p          = p
        self.k          = k
        self.varx       = varx
        self.varv       = varv
        self.device     = device
        self.x_dim      = x_dim
        self.normalizer = self.get_normalizer()
    @abc.abstractmethod
    def g2P10(self, t):
        '''
        1-0 element of the Solution of Lyapunov function P matrix.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def g2P11(self, t):
        '''
        1-1 element of the Solution of Lyapunov function P matrix.
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def sigmaxx(self, t):
        '''
        Covariance matrix xx component
        '''
        raise NotImplementedError

    @abc.abstractmethod
    def sigmavx(self, t):
        '''
        Covariance matrix xv component
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def sigmavv(self, t):
        '''
        Covariance matrix vv component
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def g(self, t):
        '''
        diffusion coefficient, can be time variant or time invariant
        '''
        raise NotImplementedError
    
    @abc.abstractmethod
    def score(self, t):
        raise NotImplementedError
        
    @abc.abstractmethod
    def get_analytic_mux_muv(self, t,x0,v0,x1):
        '''
        compute the random variance of x and v at time t given initial x0,v0 and target x1
        '''        
        raise NotImplementedError

    @abc.abstractmethod
    def mux0_muv0(self):
        '''
        mean of position and velocity at initial boundary
        '''        
        raise NotImplementedError

    @abc.abstractmethod
    def get_normalizer(self):
        raise NotImplementedError
    
    def get_cov(self,t,damp=0):
        '''
        Compute cholesky decomposition complenent Lxx,Lxv,Lvv and ell
        '''    
        x_dim       = self.x_dim
        t           = t.double()
        sigxx       = cast_shape(self.sigmaxx(t),x_dim)+damp
        sigxv       = cast_shape(self.sigmavx(t),x_dim)
        sigvv       = cast_shape(self.sigmavv(t),x_dim)+damp
        ellt        = cast_shape(-torch.sqrt(sigxx/(sigxx*sigvv-sigxv**2)),x_dim)
        Lxx         = torch.sqrt(sigxx)
        Lxv         = sigxv/Lxx
        tmp         = sigvv-Lxv**2
        invalid_idx = torch.logical_and(tmp<0, torch.isclose(tmp,torch.zeros_like(tmp)))
        tmp[invalid_idx]\
                    = 0
        Lvv         = torch.sqrt(tmp)

        return Lxx.float(),Lxv.float(),Lvv.float(), ellt.float()

    def get_xt_vt_fv(self,t,x1,DE_type,device,return_fv=True):
        '''
        Compute the input and label for the network
        '''    
        # opt                 = self.opt
        joint_dim           = [value*2 if idx==0 else value for idx,value in enumerate(self.x_dim)]
        batch_x             = t.shape[0]
        mux0,muv0           = self.mux0_muv0(batch_x)
        muxt,muvt           = self.get_analytic_mux_muv(t,mux0,muv0,x1=x1)
        Lxx,Lxv,Lvv,ell     = self.get_cov(t)
        noise       = torch.randn(batch_x, *joint_dim,device=device) 
        assert noise.shape[0] == t.shape[0]
        epsxx,epsvv = torch.chunk(noise,2,dim=1) 
        _g2P10 = self.g2P10(t)
        _g2P11 = self.g2P11(t)
        analytic_xs = muxt+Lxx*epsxx 
        analytic_vs = muvt+(Lxv*epsxx+Lvv*epsvv) 
        if return_fv: 
            normalization   = self.normalizer(t)
            analytic_fv     = 4*x1*(t-1)**2- _g2P11*((Lxx/(1-t)+Lxv)*epsxx+Lvv*epsvv)

            score           = self.score(t,ell,epsvv) if DE_type=='probODE' else 0

            analytic_fv     = (analytic_fv+score)/normalization
            # =========normlaize the label to standard gaussian =================

            return analytic_xs, analytic_vs, analytic_fv
        
        else:
            return analytic_xs, analytic_vs




class TVgMPC(BaseDynamics):
    '''
    TIVg: Time Variant g
    '''
    def __init__(self, p,k,varx,varv,x_dim,device,DE_type):
        super(TVgMPC,self).__init__(p,k,varx,varv,x_dim,device)
        self.DE_type    = DE_type
    def g2P10(self, t):
        return  6/(-1+t)**2

    def g2P11(self, t):
        return -4/(-1+t)
    
    def g(self, t):
        tt    = 1
        return self.p*(tt-t)
    
    def sigmaxx(self, t):
        m,n   = self.varx, self.varv
        k     = self.k
        p     = self.p
        tt    = 1
        val =(t - 1)**2*(30*m*(t**3 - 3*t**2 + 3*t + 3)**2 - 60*p**2*(t - 1)**3*torch.log(1 - t) - t*(60*k*np.sqrt(m*n)*(t**5 - 6*t**4 + 15*t**3 - 15*t**2 + 9) - 30*n*t*(t**2 - 3*t + 3)**2 + p**2*(t**5*(6*tt**2 + 3*tt + 1) - 6*t**4*(6*tt**2 + 3*tt + 1) + 15*t**3*(6*tt**2 + 3*tt + 1) - 10*t**2*(9*tt**2 + 11) + 150*t - 60)))/270
        return val

    def sigmavx(self, t):
        m,n   = self.varx, self.varv
        p     = self.p
        k     = self.k
        tt    = 1
        val =(1/270 - t/270)*(30*k*np.sqrt(m*n)*(8*t**6 - 48*t**5 + 120*t**4 - 135*t**3 + 45*t**2 + 27*t - 9) + 150*p**2*(t - 1)**3*torch.log(1 - t) + t*(-120*m*(t**5 - 6*t**4 + 15*t**3 - 15*t**2 + 9) - 30*n*(4*t**5 - 24*t**4 + 60*t**3 - 75*t**2 + 45*t - 9) + p**2*(4*t**5*(6*tt**2 + 3*tt + 1) - 24*t**4*(6*tt**2 + 3*tt + 1) + 60*t**3*(6*tt**2 + 3*tt + 1) - 5*t**2*(81*tt**2 + 18*tt + 55) + 15*t*(9*tt**2 + 25) - 150)))
        return val


    
    def sigmavv(self, t):
        m,n   = self.varx, self.varv
        p     = self.p
        k     = self.k
        tt    = 1
        val= n*(-4*t**3 + 12*t**2 - 12*t + 3)**2/9 - 8*p**2*(t - 1)**3*torch.log(1 - t)/9 + t*(-120*k*np.sqrt(m*n)*(4*t**5 - 24*t**4 + 60*t**3 - 75*t**2 + 45*t - 9) + 240*m*t*(t**2 - 3*t + 3)**2 + p**2*(-8*t**5*(6*tt**2 + 3*tt + 1) + 48*t**4*(6*tt**2 + 3*tt + 1) - 120*t**3*(6*tt**2 + 3*tt + 1) + 5*t**2*(180*tt**2 + 72*tt + 53) - 15*t*(36*tt**2 + 9*tt + 20) + 135*tt**2 + 120))/135
        return val
    
    def get_normalizer(self):
        '''
        get Compute the normlaizier for the label in order to normalize the label to some range.
        '''    
        return self.normalizer
        
    def normalizer(self,t):
        '''
        normlaize the label to [1-->0] (predicting the x1) or standard gaussian (predicting the noise)
        '''
        _g2P10 = self.g2P10(t)
        _g2P11 = self.g2P11(t)
        _g     = self.g(t)
        Lxx,Lxv,Lvv,ell     = self.get_cov(t)
        Lxx = Lxx.reshape_as(t)
        Lxv = Lxv.reshape_as(t)
        Lvv = Lvv.reshape_as(t)
        ell = ell.reshape_as(t)
        # if len(t.shape)==4:
        #     debug()

        if self.DE_type=='probODE':
            norm =torch.sqrt(((_g2P11*((-1/(1-t)*Lxx-Lxv)))**2+(_g2P11*Lvv-0.5*_g**2*ell)**2))/(1-t)
            return norm
        else:
            damp=0
            return torch.sqrt(((_g2P11*((-1/(1-t)*Lxx-Lxv)))**2+(_g2P11*Lvv)**2)+damp)/(1-t)


    def get_analytic_mux_muv(self, t,x0,v0,x1):
        bs = x0.shape[0]
        if x1 is None:
            return self.mux0_muv0(bs)
        else:
            muv     =v0*(-4*t**3/3 + 4*t**2 - 4*t + 1) + t*(-4*x0/3 + 4*x1/3)*(t**2 - 3*t + 3)
            mux     =-x0*(t**4 - 4*t**3 + 6*t**2 - 3)/3 + t*(-v0*(t**3 - 4*t**2 + 6*t - 3) + x1*t*(t**2 - 4*t + 6))/3
            return mux,muv
    
    def mux0_muv0(self,bs):
        muv0    = torch.zeros(bs, *self.x_dim,device=self.device)
        mux0    = torch.zeros(bs, *self.x_dim,device=self.device)
        return mux0, muv0    
        
    def score(self,t,ell,epsvv):
        _g  = self.g(t)
        return -0.5*_g**2*ell*epsvv

    def get_m0(self,bs):
        joint_dim           = [value*2 if idx==0 else value for idx,value in enumerate(self.x_dim)]

        mux0,muv0           = self.mux0_muv0(bs)
        t                   = torch.zeros(bs,1,device=self.device)
        Lxx,Lxv,Lvv,ell     = self.get_cov(t)
        noise       = torch.randn(bs, *joint_dim,device=self.device) 
        assert noise.shape[0] == t.shape[0]
        epsxx,epsvv = torch.chunk(noise,2,dim=1) 
        analytic_x0 = mux0+Lxx*epsxx 
        analytic_v0 = muv0+(Lxv*epsxx+Lvv*epsvv) 
        return torch.cat([analytic_x0, analytic_v0],dim=1)