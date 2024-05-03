#**********Toy******************
#SSS Stochastic dynamics
#train
python train.py --name train-toy/sde-spiral      --exp toy --toy-exp spiral 
#sampling
python train.py --name train-toy/sde-spiral-eval --exp toy --toy-exp spiral --eval --ckpt train-toy/sde-spiral --nfe 10
#Exponential Integrator ode dynamics
# train
python train.py --name train-toy/ode-spiral      --exp toy --toy-exp spiral --DE-type probODE --solver gDDIM
#Sampling
python train.py --name train-toy/ode-spiral-eval --exp toy --toy-exp spiral --DE-type probODE --solver gDDIM --nfe 10 --eval --ckpt train-toy/ode-spiral/latest.pt
#**************Cifar10*************
#NFE=5 FID=11.88
CUDA_VISIBLE_DEVICES=1 python sampling.py  --n-gpu-per-node 1  --ckpt Remote_Cifar10_ODE --pred-x1 --solver gDDIM  --T 0.4 --nfe 5 --fid-save-name cifar10-nfe5 --port 6024  --num-sample 50000 --batch-size 1000
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe5 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#NFE=10 FID=4.54
CUDA_VISIBLE_DEVICES=1 python sampling.py  --n-gpu-per-node 1  --ckpt Remote_Cifar10_ODE --pred-x1 --solver gDDIM  --T 0.7 --nfe 10 --fid-save-name cifar10-nfe10 --port 6024  --num-sample 50000 --batch-size 1000
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe10 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#NFE=20 FID=2.58
CUDA_VISIBLE_DEVICES=0 python sampling.py  --n-gpu-per-node 1  --ckpt Remote_Cifar10_ODE --pred-x1 --solver gDDIM  --T 0.9 --nfe 20 --fid-save-name cifar10-nfe20 --port 6024  --num-sample 50000 --batch-size 1000 
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe20 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#**************AFHQv2*************
#Conditional Generation
python sampling.py  --n-gpu-per-node 1  --ckpt AFHQ-cond/uuysqcynde/latest.pt  --solver gDDIM  --save-img  --img-save-name cond-test0129 --nfe 100 --T 0.999 --num-sample 8 --batch-size 8 --stroke-path dataset/StrokeData/testFig0.png --stroke-type dyn-v 
python sampling.py  --n-gpu-per-node 1  --ckpt AFHQ-cond/uuysqcynde/latest.pt  --solver gDDIM  --save-img  --img-save-name cond-test0129 --nfe 100 --T 0.999 --num-sample 8 --batch-size 8 --stroke-path dataset/StrokeData/testFig1.png --stroke-type dyn-v 
#Impainting Generation
python sampling.py  --n-gpu-per-node 1  --ckpt Remote_AFHQv2_ODE  --pred-x1 --solver gDDIM  --save-img  --img-save-name impaint-AFHQv2 --nfe 100 --T 0.999 --num-sample 64 --batch-size 64 --stroke-type dyn-v --impainting --stroke-path dataset/StrokeData/testFig0_impainting.png
python sampling.py  --n-gpu-per-node 1  --ckpt Remote_AFHQv2_ODE  --pred-x1 --solver gDDIM  --save-img  --img-save-name impaint-AFHQv2 --nfe 100 --T 0.999 --num-sample 64 --batch-size 64 --stroke-type dyn-v --impainting --stroke-path dataset/StrokeData/testFig1_impainting.png

#NFE=5
CUDA_VISIBLE_DEVICES=1 python sampling.py  --n-gpu-per-node 1  --ckpt Remote_Cifar10_ODE --pred-x1 --solver gDDIM  --T 0.4 --nfe 5 --fid-save-name cifar10-nfe5 --port 6024  --num-sample 50000 --batch-size 1000
torchrun --standalone --nproc_per_node=1 fid.py calc --images=FID_EVAL/cifar10-nfe5 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe5 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#NFE=10
CUDA_VISIBLE_DEVICES=1 python sampling.py  --n-gpu-per-node 1  --ckpt cifar10_ODE/latest.pt --pred-x1 --solver gDDIM  --T 0.7 --nfe 10 --fid-save-name cifar10-nfe10 --port 6024  --num-sample 50000 --batch-size 1000
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe10 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#NFE=20
CUDA_VISIBLE_DEVICES=0 python sampling.py  --n-gpu-per-node 1  --ckpt cifar10_ODE/latest.pt --pred-x1 --solver gDDIM  --T 0.9 --nfe 20 --fid-save-name cifar10-nfe20 --port 6024  --num-sample 50000 --batch-size 1000 
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe20 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#**************ImageNet*************
#NFE=5
CUDA_VISIBLE_DEVICES=1 python sampling.py  --n-gpu-per-node 1  --ckpt Remote_Cifar10_ODE --pred-x1 --solver gDDIM  --T 0.4 --nfe 5 --fid-save-name cifar10-nfe5 --port 6024  --num-sample 50000 --batch-size 1000
torchrun --standalone --nproc_per_node=1 fid.py calc --images=FID_EVAL/cifar10-nfe5 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe5 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#NFE=10
CUDA_VISIBLE_DEVICES=1 python sampling.py  --n-gpu-per-node 1  --ckpt cifar10_ODE/latest.pt --pred-x1 --solver gDDIM  --T 0.7 --nfe 10 --fid-save-name cifar10-nfe10 --port 6024  --num-sample 50000 --batch-size 1000
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe10 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

#NFE=20
CUDA_VISIBLE_DEVICES=0 python sampling.py  --n-gpu-per-node 1  --ckpt cifar10_ODE/latest.pt --pred-x1 --solver gDDIM  --T 0.9 --nfe 20 --fid-save-name cifar10-nfe20 --port 6024  --num-sample 50000 --batch-size 1000 
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe20 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 edm/fid.py calc --images=FID_EVAL/cifar10-nfe20 --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

# https://drive.google.com/file/d/1H92Bgz26hLajYNtcY7zPtI7y9HFLeYtD/view?usp=sharing
# https://drive.google.com/file/d/1u26_iWWaBSW8hXMnAudJB90Awolabta4/view?usp=sharing