#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import sys
sys.path.append("..") # Adds higher directory to python modules path.
from AM import util
def tmp_fnc(t):
    '''Known issue for transforms in DDP settting'''
    return (t * 2) - 1

class cifar10_data():
    """cifar10 dataset."""
    def __init__(self, opt):
        self.opt    = opt
        bs          = opt.microbatch

        x1          = self.generate_x1()
        self.loader = util.setup_loader(x1,bs)

    def generate_x1(self):
        transforms_list = [transforms.RandomHorizontalFlip(p=0.5)] if self.opt.xflip else []
        transforms_list+=[
                    transforms.ToTensor(), #Convert to [0,1]
                    transforms.Lambda(tmp_fnc) #Convert to [-1,1]
                ]
        x1=datasets.CIFAR10(
                './dataset',
                train= True,
                download=True,
                transform=transforms.Compose(transforms_list)
            )
        return x1

    def sample(self):
        x1,label    = next(self.loader)[0],next(self.loader)[1]
        label       =torch.nn.functional.one_hot(label, num_classes=10)
        return x1.to(self.opt.device),label.to(self.opt.device) if self.opt.cond else None
