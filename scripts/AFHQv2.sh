# python sampling.py  --n-gpu-per-node 8 --batch-size 1250 --ckpt AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/latest.pt  --num-sample 50000 --sampling sscs --pred-x1 --nfe 20
# torchrun --standalone --nproc_per_node=1 fid.py calc --images=results/AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/fid_sample_folder/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz

python sampling.py  --n-gpu-per-node 8 --batch-size 1250 --ckpt AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/latest.pt  --num-sample 50000 --sampling sscs --pred-x1 --nfe 50
torchrun --standalone --nproc_per_node=1 fid.py calc --images=results/AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/fid_sample_folder/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz

python sampling.py  --n-gpu-per-node 8 --batch-size 1250 --ckpt AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/latest.pt  --num-sample 50000 --sampling sscs --pred-x1 --nfe 100
torchrun --standalone --nproc_per_node=1 fid.py calc --images=results/AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/fid_sample_folder/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz

python sampling.py  --n-gpu-per-node 8 --batch-size 1250 --ckpt AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/latest.pt  --num-sample 50000 --sampling sscs --pred-x1 --nfe 150
torchrun --standalone --nproc_per_node=1 fid.py calc --images=results/AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/fid_sample_folder/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz

python sampling.py  --n-gpu-per-node 8 --batch-size 1250 --ckpt AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/latest.pt  --num-sample 50000 --sampling sscs --pred-x1 --nfe 500
torchrun --standalone --nproc_per_node=1 fid.py calc --images=results/AFHQv2-uniform-recip-precond-bs512-varv12-k0.8-p3-lr5e4-EDM-newlabel/fid_sample_folder/ --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/afhqv2-64x64.npz
