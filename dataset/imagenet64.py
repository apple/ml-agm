#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
"""Streaming images and labels from datasets created with dataset_tool.py."""
from edm.dataset import ImageFolderDataset
from AM import util
try:
    import pyspng
except ImportError:
    pyspng = None

class imagenet64_data():
    """cifar10 dataset."""
    def __init__(self, opt):
        self.opt    = opt
        bs          = opt.microbatch
        x1          = ImageFolderDataset(path='dataset/imagenet-64x64.zip',use_labels=opt.cond,xflip=opt.xflip)
        self.loader = util.setup_loader(x1,bs,num_workers=opt.n_gpu_per_node)

    def sample(self):
        x1,label    = next(self.loader)[0],next(self.loader)[1] #[bs, dims],[bs,one_hot]
        x1          = x1/ 127.5 - 1
        return x1.to(self.opt.device),label.to(self.opt.device) if self.opt.cond else None
#----------------------------------------------------------------------------
# Abstract base class for datasets.
