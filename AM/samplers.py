#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
from tqdm import tqdm
import torch
from . import util
import numpy as np


class SamplerWrapper:
    def __init__(self,**kwargs):
        solver_name = kwargs['solver_name']
        diz         = kwargs['diz']
        t0          = kwargs['t0']
        T           = kwargs['T']
        interval    = kwargs['interval']
        device      = kwargs['device']
        dyn         = kwargs['dyn']
        diz_order   = kwargs['diz_order']
        cond_opt    = kwargs['cond_opt']
        self.kwargs = kwargs

        self.sampler            = get_solver_fn(solver_name)
        tsdts                   = get_discretizer(diz,t0,T,interval,device,diz_order)
        self.kwargs['ts_dts']   = tsdts
        DE_type                 = 'probODE'
        self.dyn                = dyn

        remove_keys=['diz','solver_name','t0','T','interval','device','diz_order']
        if solver_name=='gDDIM':
            self.r                  = self.kwargs['gDDIM_r']
            ts,dts                  = tsdts
            coef                    = AB_fn(dyn.normalizer,DE_type,ts,dts,self.r)
            self.kwargs['coef']     = coef
            self.kwargs['cond_opt'] = cond_opt
        else:
            remove_keys+=['gDDIM_r']
            remove_keys+=['cond_opt']
            
        for key in remove_keys:
            del self.kwargs[key]
        

    def solve(self,ema,net,m0,cond=None):
        with ema.average_parameters():
            return self.sampler(m0,net,cond,**self.kwargs)


def get_est_x1(dyn,t,_g,_fv,x,v,DE_type):
    _g2P11=dyn.g2P11(t)
    if DE_type=='probODE':
        Lxx,Lxv,Lvv, ell\
                    = dyn.get_cov(t)
        AA      = (Lxx/(1-t)+Lxv).to(torch.float64)
        BB      = (Lvv+(0.5*_g**2*ell)/_g2P11).to(torch.float64)
        bb      = BB/Lvv
        aa      = (AA-bb*Lxv)/Lxx
        cv      = 4/3*t*(3+(-3+t)*t)
        cx      = 1/3*t**2*(6+(-4+t)*t)
        est_x1  = (_fv+_g2P11*(aa*x+bb*v))/(4*(t-1)**2+_g2P11*(aa*cx+bb*cv))
    else:
        est_x1 = (_fv/_g2P11+v)*(1-t)+x
    return est_x1

def get_solver_fn(solver_name):
    if solver_name == 'sscs':
        return sscs_sampler
    elif solver_name == 'em':
        return em_sampler
    elif solver_name == 'gDDIM':
        return gDDIM_sampler
    else:
        raise NotImplementedError(
            'Sampler %s is not implemened.' % solver_name)


def get_discretizer(diz,t0,T,interval,device,diz_order=2):
    if diz =='Euler':
        ts  = torch.linspace(t0, T, interval+1, device=device)
        dts = ts[1:]-ts[0:-1]
        ts  = ts[0:-1]
        last_dt=torch.Tensor([0.999-T]).to(device) #For evaluate full timesteps
        dts = torch.cat([dts,last_dt],dim=0)               

    elif diz =='quad':
        ts= torch.linspace(t0**2,T**2,interval+1)
        ts= torch.sqrt(ts)
        dts= ts[1:]-ts[0:-1]
        ts      = ts[0:-1]
        ts      = ts.to(device)
        dts     = dts.to(device)

    elif diz =='rev-quad':
        order = diz_order
        ts= torch.linspace(t0**(1/order),T**(1/order),interval,dtype=torch.float64,device=device)
        ts= ts**order
        dts= ts[1:]-ts[0:-1]
        ts      = ts
        last_dt=torch.Tensor([0.999-T]).to(device) #For evaluate full timesteps
        dts = torch.cat([dts,last_dt],dim=0)
    else:
        raise NotImplementedError(
            'discretizer %s is not implemened.' % diz)
    return ts,dts
    

def dw(x,dt):
    return torch.randn_like(x)*torch.sqrt(dt)


def sscs_sampler(m0,drift,cond,ts_dts,dyn,snap,local_rank,return_est_x1=True):
    #Equivalence reduced variance
    def sigmaxx(p,t):
        return  p**2*t**3*(t*(t - 5) + 10)/30
    def sigmavx(p,t):
        return  p**2*t**2*(t*(t - 4) + 6)/12
    def sigmavv(p,t):
        return p**2*t*(t*(t - 3) + 3)/3

    def analytic_dynamics(m,t,dt):
        dt=dt/2
        delta_varxx = (sigmaxx(dyn.p*(1-t),dt)).reshape(-1,*([1,]*(len(m.shape)-1)))
        delta_varxv = (sigmavx(dyn.p*(1-t),dt)).reshape(-1,*([1,]*(len(m.shape)-1)))
        delta_varvv = (sigmavv(dyn.p*(1-t),dt)).reshape(-1,*([1,]*(len(m.shape)-1)))
        cholesky11 = torch.sqrt(delta_varxx)
        cholesky21 = (delta_varxv / cholesky11)
        cholesky22 = (torch.sqrt(delta_varvv - cholesky21 ** 2.))
        batch_randn = torch.randn_like(m, device=m.device)
        batch_randn_x, batch_randn_v = torch.chunk(batch_randn, 2, dim=1)
        noise_x = cholesky11 * batch_randn_x
        noise_v = cholesky21 * batch_randn_x + cholesky22 * batch_randn_v
        noise = torch.cat((noise_x, noise_v), dim=1)
        x,v=torch.chunk(m,2,dim=1)
        x = x+v*dt
        m = torch.cat([x,v],dim=1)
        perturbed_data = m +noise
        return perturbed_data       

    def EM_dynamics(v,dyn,fv,t,normalizer):
        norm        = (normalizer(t)).squeeze()
        fv          = fv*norm
        v           = v+fv*dt
        return v,fv        

    assert dyn.DE_type == 'SDE'
    bs      = m0.shape[0]
    ts,dts  = ts_dts
    m0      = m0.to(torch.float64)
    m       = m0
    ms      = []
    x,v     = torch.chunk(m0,2,dim=1)
    interval = ts.shape[0]
    snaps   = np.linspace(0, interval-1, snap).astype(int)
    snapts  = []
    if local_rank == 0:
        _ts     = tqdm(ts,desc=util.yellow("Propagating Dynamics..."))
    else:
        _ts = ts
    m       = m0
    ms      = []
    snapts  = []
    est_x1s = []
    normalizer=\
            dyn.get_normalizer()
    x,v = torch.chunk(m0,2,dim=1)
    for idx,(t,dt) in enumerate(zip(_ts,dts)):
        _t= t.repeat(bs)
        m = analytic_dynamics(m,_t,dt)
        x,v         = torch.chunk(m,2,dim=1)
        
        fv          = drift(m.to(torch.float32),_t.to(torch.float32),cond=cond).to(torch.float32)
        #============EM step=============
        v,_fv       = EM_dynamics(v,dyn,fv,t,normalizer)
        #============EM step=============
        m       = torch.cat([x,v],dim=1)       
        m = analytic_dynamics(m,_t,dt)

        if idx in snaps: 
            ms.append(m[:,None,...])
            snapts.append(t[None])
            if return_est_x1:
                _g2P11=dyn.g2P11(t)
                est_x1 = (_fv/_g2P11+v)*(1-t)+x 
                est_x1s.append(est_x1[:,None,...])

    xT  = x+v*dt
    mT  = torch.cat([xT,v],dim=1)
    ms.append(mT[:,None,...])
    if return_est_x1: est_x1s.append(est_x1[:,None,...])
    snapts.append(t[None])
    return torch.cat(ms,dim=1),mT, torch.cat(est_x1s,dim=1), torch.cat(snapts,dim=0)


def em_sampler(m0,drift,cond,ts_dts,dyn,snap,local_rank,return_est_x1=True):
    DE_type = dyn.DE_type
    bs      = m0.shape[0]
    rank    = local_rank
    ts,dts  = ts_dts
    interval= ts.shape[0]
    if rank == 0:
        times_horizon = zip(tqdm(ts,desc=util.blue("Propagating Dynamics..."),position=0,leave=False,colour='blue'),dts)
    else:
        times_horizon=zip(ts,dts)
    m       = m0
    ms      = []
    x,v     = torch.chunk(m0,2,dim=1)
    snaps   = np.linspace(0, interval-1, snap).astype(int)
    snapts  = []
    normalizer=\
            dyn.get_normalizer()
    if return_est_x1: est_x1s     = []

    for idx,(t,dt) in enumerate(times_horizon):
        _t          = t.repeat(bs)
        _g          = dyn.g(t)
        fv          = drift(m,_t,cond)
        _g2P11      = dyn.g2P11(t)

        norm        = (normalizer(t)).squeeze()
        fv          = fv*norm
        m   = torch.cat([x,v],dim=1)

        #=========dyn propagation===========
        x       = x+v*dt
        dw = dw(v,dt) if DE_type == 'SDE' else torch.zeros_like(v)
        v       = v+fv*dt+_g*dw
        #=========dyn propagation===========

        if idx in snaps: 
            ms.append(m[:,None,...])
            snapts.append(t[None])
            if return_est_x1:
                est_x1 = get_est_x1(dyn,t,_g,fv,x,v,DE_type)
                est_x1s.append(est_x1[:,None,...])

    mT = m

    ms.append(mT[:,None,...])

    est_x1s.append(est_x1[:,None,...])
    snapts.append(t[None])
    return torch.cat(ms,dim=1),mT, torch.cat(est_x1s,dim=1), torch.cat(snapts,dim=0)




@torch.no_grad()
def gDDIM_sampler(m0,drift,cond,ts_dts,dyn,coef,snap,gDDIM_r,local_rank,return_est_x1=True,cond_opt=None):
    ts,dts      = ts_dts
    conf_flag   = False if cond_opt is None else True


    DE_type = dyn.DE_type
    assert DE_type == 'probODE'
    r       = gDDIM_r
    rank    = local_rank
    bs      = m0.shape[0]
    m0      = m0.to(torch.float64)
    m       = m0
    ms      = []
    x,v     = torch.chunk(m0,2,dim=1)
    interval = ts.shape[0]
    snaps   = np.linspace(0, interval-1, snap).astype(int)
    snapts  = []
    normalizer=\
            dyn.get_normalizer()
    intgral_norm = coef

    if rank == 0:
        times_horizon = zip(tqdm(ts,desc=util.blue("Propagating Dynamics..."),position=0,leave=False,colour='blue'),dts)
    else:
        times_horizon=zip(ts,dts)

    if return_est_x1: est_x1s     = []
    prev_fv = []

    
    if conf_flag:
        stroke      = cond_opt.stroke
        stroke_type = cond_opt.stroke_type
        impainting  = cond_opt.impainting
        cond_strength\
                    = 1.0 if impainting else 0.25
        cond_fn     = impaint_stroke if impainting else dyn_stroke
        if stroke_type=='dyn-v':  stroke_idx   = int(cond_strength*ts.shape[0])
        if stroke_type=='init-v': v            = 0.9*v+0.1*stroke

    for idx,(t,dt) in enumerate(times_horizon):
        _t          = t.repeat(bs)
        _g          = dyn.g(t)
        normt       = (normalizer(t)).squeeze()

    # =======Conditional generation ==========
        if conf_flag and idx==0:
            fv      = drift(m.to(torch.float32),_t.to(torch.float32),cond=cond).to(torch.float32)
            _fv     = fv*normt
            est_x1  = get_est_x1(dyn,t,_g,_fv,x,v,DE_type)

        if conf_flag and stroke_type=='dyn-v' and idx<stroke_idx: v=cond_fn(t,dyn,stroke,m,est_x1)

        m           = torch.cat([x,v],dim=1)
    # =======Conditional generation ==========

    #=========dyn propagation===========
        fv      = drift(m.to(torch.float32),_t.to(torch.float32),cond=cond).to(torch.float32)

        if idx in snaps: 
            _fv = fv*normt

            ms.append(m[:,None,...])
            snapts.append(t[None])
            if return_est_x1:
                est_x1  = get_est_x1(dyn,t,_g,_fv,x,v,DE_type)
                est_x1s.append(est_x1[:,None,...])

        accumulated_fx = 0
        accumulated_fv = 0
        max_order = min(idx,r)
        for jj in range(max_order+1):
            coef = intgral_norm[idx][jj]
            if jj==0:
                if DE_type=='ODE':
                    coef_fv = coef*fv/(1-t)
                else:
                    coef_fx = coef[0,1]*fv
                    coef_fv = coef[1,1]*fv
            else:
                # print(jj)
                assert jj-1>=0 and len(prev_fv)==max_order

                if DE_type=='ODE':
                    coef_fv = prev_fv[jj-1]/(1-t)*coef
                else:
                    coef_fx = prev_fv[jj-1]*coef[0,1]
                    coef_fv = prev_fv[jj-1]*coef[1,1]

            accumulated_fv += coef_fv
            accumulated_fx += coef_fx


        phit    = phi_fn((t+dt)[None,None],t[None,None])[0]
        x       = phit[0,0]*x+phit[0,1]*v+accumulated_fx
        v       = phit[1,0]*x+phit[1,1]*v+accumulated_fv
        m       = torch.cat([x,v],dim=1)
        if len(prev_fv)<r:
            prev_fv.insert(0,fv)
        elif len(prev_fv)==0:
            pass
        else:
            prev_fv.pop(-1)
            prev_fv.insert(0,fv)

    mT = m
    ms.append(mT[:,None,...])
    est_x1s.append(est_x1[:,None,...])
    snapts.append(t[None])

    return torch.cat(ms,dim=1),mT, torch.cat(est_x1s,dim=1), torch.cat(snapts,dim=0)


#========Preparing the numerical integration by monte carlo sampling==========
def monte_carlo_integral(fn,t0,t1,num_sample):
    ts = torch.linspace(t0,t1,num_sample, device=t0.device)[:,None]
    return (fn(ts)).sum(0)*(t1-t0)/num_sample

def phi_fn(t,s):
    '''
    s<t
    '''
    num_dts = 1000
    phi     = torch.eye(2)
    phis    = torch.zeros(s.shape[0],2,2)
    phis[:,0,0]=1
    phis[:,0,1]=(t-s)[:,0]
    phis[:,1,1]=1
    return phis
    
def extrapolate_fn(normalizer,DE_type, ts,dts,i,j,r):
    def _fn_r(t):
        device=t.device
        z   = torch.zeros(t.shape[0],2,2)
        z[:,1,1]=normalizer(t)[:,0]
        z   = z.to(device)
        #=====time coef=========
        prod= torch.eye(2,device=device)
        for k in range(r+1):
            assert i-k>=0 and i-j>=0
            if k!=j:
                prod= prod* ((t-ts[i-k])/(ts[i-j]-ts[i-k]))[...,None]
        #=====time coef=========
        phi_matrix = phi_fn(ts[i]+dts[i],t).to(device)
        if DE_type=='ode':
            return phi_matrix@z@prod*(1-t)
        else:
            return phi_matrix@z@prod
    return _fn_r

def AB_fn(normalizer,DE_type,ts,dts,r=0):
    intgral_norm = {}
    num_monte_carlo_sample = 50000
    for idx,(t,dt) in enumerate(zip(ts,dts)):
        max_order       = min(idx,r)
        intgral_norm[idx] = {}
        for jj in range(max_order+1):
            coef_fn         = extrapolate_fn(normalizer,DE_type,ts,dts,idx,j=jj,r=max_order)
            coef            = monte_carlo_integral(coef_fn,t,t+dt,num_monte_carlo_sample)
            intgral_norm[idx][jj]=coef
    return intgral_norm


def dyn_stroke(t,dyn,stroke,m,est_x1=None):

    currx,currv = torch.chunk(m,2,dim=1)
    Sigxx,Sigxv,Sigvv\
                = dyn.sigmaxx(t),dyn.sigmavx(t),dyn.sigmavv(t)
    noise       = torch.randn(*m.shape,device=m.device) 
    epsxx,epsvv = torch.chunk(noise,2,dim=1)     
    muv         = 0*(-4*t**3/3 + 4*t**2 - 4*t + 1) + t*(-4*0/3 + 4*stroke/3)*(t**2 - 3*t + 3)
    mux         = -0*(t**4 - 4*t**3 + 6*t**2 - 3)/3 + t*(-0*(t**3 - 4*t**2 + 6*t - 3) + stroke*t*(t**2 - 4*t + 6))/3
    muv         = muv+Sigxv/Sigxx*(currx-mux)
    Sigv        = Sigvv-Sigxv**2/Sigxx
    v           = muv+Sigv*epsvv
    return v

def impaint_stroke(t,dyn,stroke,m,est_x1):
    mask        = torch.ones_like(stroke)
    mask[stroke==-1]\
                = 0
    invmask     = 1-mask 
    stroke      = stroke*mask+invmask*est_x1
    return dyn_stroke(t,dyn,stroke,m,est_x1)

