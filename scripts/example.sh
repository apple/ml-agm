#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
#Cifar10
python sampling.py  --n-gpu-per-node 1  --ckpt Cifar10-ODE/latest.pt --pred-x1 --solver gDDIM  --T 0.9 --nfe 20 --fid-save-name cifar10-nfe20  --num-sample 64 --batch-size 64 --save-img --img-save-name cifar10-nfe20
#AFHQv2
python sampling.py  --n-gpu-per-node 1  --ckpt AFHQv2-ODE/latest.pt --pred-x1 --solver gDDIM  --T 0.9 --nfe 20 --fid-save-name AFHQv2-nfe20  --num-sample 64 --batch-size 64 --save-img --img-save-name AFHQv2-nfe20

#AFHQv2 Conditional Generation
python sampling.py  --n-gpu-per-node 1  --ckpt AFHQv2-ODE/latest.pt  --pred-x1 --solver gDDIM  --save-img  --img-save-name cond-AFHQv2 --nfe 100 --T 0.999 --num-sample 64 --batch-size 64 --stroke-path dataset/StrokeData/testFig0.png --stroke-type dyn-v # you can also replace it by init-v

#Imagenet64
#NFE 20 FID=10.55
python sampling.py  --n-gpu-per-node 1  --ckpt uncond-ImageNet64-ODE/latest.pt  --pred-x1 --solver gDDIM  --save-img --img-save-name imagenet-nfe20 --fid-save-name ImageNet64-nfe20 --nfe 20 --T 0.99    --num-sample 64 --batch-size 64

