#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""Streaming images and labels from datasets created with dataset_tool.py."""
# /home/iamctr/Desktop/ACDS/AM/playground/refac-CG/AM-dev/dataset/AFHQv2.py
from edm.dataset import ImageFolderDataset
# /home/iamctr/Desktop/ACDS/AM/playground/refac-CG/AM-dev/edm/dataset.py
from AM import util
try:
    import pyspng
except ImportError:
    pyspng = None

class AFHQv2_data():
    """cifar10 dataset."""
    def __init__(self, opt):
        self.opt    = opt
        bs          = opt.microbatch
        x1          = ImageFolderDataset(path='dataset/afhqv2-64x64.zip')
        self.loader = util.setup_loader(x1,bs,num_workers=opt.n_gpu_per_node)
        assert not opt.cond

    def sample(self):
        x1      = next(self.loader)[0]
        x1      = x1/ 127.5 - 1
        label   = None
            
        return x1.to(self.opt.device),label
