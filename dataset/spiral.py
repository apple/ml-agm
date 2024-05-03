#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.datasets import make_swiss_roll

class spiral_data(Dataset):
    """Toy Spiral dataset."""
    def __init__(self, opt):
        n_train     = opt.n_train
        self.opt    = opt
        self.x1     = self.generate_x1(n_train)
        self.bs     = opt.microbatch
    def generate_x1(self,n):
        '''
        n: number of total samples
        '''
        if self.opt.toy_exp=='gmm':
            WIDTH = 3
            BOUND = 0.5
            NOISE = 0.04
            ROTATION_MATRIX = np.array([[1., -1.], [1., 1.]]) / np.sqrt(2.)

            means = np.array([(x, y) for x in np.linspace(-BOUND, BOUND, WIDTH)
                            for y in np.linspace(-BOUND, BOUND, WIDTH)])
            means = means @ ROTATION_MATRIX
            covariance_factor = NOISE * np.eye(2)

            index = np.random.choice(
                range(WIDTH ** 2), size=n, replace=True)
            noise = np.random.randn(n, 2)
            data = means[index] + noise @ covariance_factor
            data=torch.from_numpy(data.astype('float32'))
            data=data.to(self.opt.device)
        elif self.opt.toy_exp=='spiral':
            NOISE = 0.3
            MULTIPLIER = 0.05
            OFFSETS = [[1.2, 1.2], [1.2, -1.2], [-1.2, -1.2], [-1.2, 1.2]]

            idx = np.random.multinomial(n, [0.2] * 5, size=1)[0]

            sr = []
            for k in range(5):
                sr.append(make_swiss_roll(int(idx[k]), noise=NOISE)[
                        0][:, [0, 2]].astype('float32') * MULTIPLIER)

                if k > 0:
                    sr[k] += np.array(OFFSETS[k - 1]).reshape(-1, 2)

            data = np.concatenate(sr, axis=0)[np.random.permutation(n)]
            data = torch.from_numpy(data.astype('float32'))
            data=data.to(self.opt.device)
        else:
            raise RuntimeError
        
        return data
    
    def __len__(self):
        return self.x1.shape[0]

    def sample(self):
        bs      = self.bs
        lenth   = self.x1.shape[0]
        idx     = torch.randint(lenth,  (bs,),device=self.opt.device)
        x1      = self.x1[idx]
        return x1, None
