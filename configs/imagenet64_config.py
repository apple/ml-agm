#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import ml_collections
def get_imagenet64_default_configs():
  config = ml_collections.ConfigDict()
  # basic
  config.training = ml_collections.ConfigDict()
  config.seed             = 42
  config.microbatch       = 64
  config.n_gpu_per_node   = 8
  config.lr               = 2e-4
  config.precond          = True
  config.reweight_type    = 'reciprocal'
  config.t_samp           = 'uniform'
  config.num_itr          = 10000000
  config.data_dim         = [3,64,64]
  config.joint_dim        = [6,64,64] 
  #data
  config.xflip            = True
  config.exp              = 'imagenet64'

  #Dynamics
  config.t0               = 1e-5
  config.T                = 0.999
  config.dyn_type         = 'TVgMPC'
  # config.algo             = 'DM'
  config.clip_grad        = 1
  config.damp_t           = 1
  config.p                = 3
  config.k                = 0.2
  config.varx             = 1
  config.varv             = 1
  config.DE_type          = 'probODE'

  #Evaluation during training
  config.nfe              = 100 #Evaluation interval during training, can be replaced
  config.solver           = 'gDDIM'
  config.diz_order        = 2
  config.diz              = 'rev-quad'
  config.train_fid_sample = 4096 #Number of sample to evaluate FID during training
  

  model_configs           = get_edm_ADM_config()
  # model_configs=get_Unet_config()
  return config, model_configs

def get_edm_ADM_config():
  config                      = ml_collections.ConfigDict()
  config.image_size           = 64
  config.name                 = "ADM"
  config.in_channels          = 6
  config.out_channels         = 3
  return config
