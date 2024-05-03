#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torchvision.datasets as datasets
datasets.CIFAR10(
        './dataset',
        train= True,
        download=True,
    )