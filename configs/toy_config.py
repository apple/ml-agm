#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import ml_collections

def get_toy_default_configs():
  config = ml_collections.ConfigDict()
  # training
  # config.training         = ml_collections.ConfigDict()
  config.seed             = 42
  config.num_itr          = 40001
  config.t0               = 1e-5
  config.debug            = True
  config.microbatch       = 2048
  config.nfe              = 200
  config.DE_type          = 'SDE'
  config.t_samp           = 'uniform'
  config.diz              = 'Euler'
  config.solver           = 'sscs'
  config.exp              = 'toy'
  config.lr               = 1e-3
  config.dyn_type         = 'TVgMPC'
  config.T                = 0.999
  config.p                = 3
  config.k                = 0.2
  config.varx             = 1
  config.varv             = 1
  config.data_dim         = [2]
  config.joint_dim        = [4]
  config.reweight_type    = 'reciprocal'

  model_configs=None
  return config, model_configs
