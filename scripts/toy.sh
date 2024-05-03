python train.py --name prob-ode/varv10-p1 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 1 --varv 3 --p 1 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# python train.py --name prob-ode/varv7 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 1 --varv 7 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# python train.py --name prob-ode/varv5 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 1 --varv 5 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# python train.py --name prob-ode/varv3 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 1 --varv 3 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# python train.py --name prob-ode/varv1 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 1 --varv 1 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode


# python train.py --name prob-ode/varx2-varv10 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 2 
# --varv 10 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# python train.py --name prob-ode/varx4-varv10 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 4 --varv 10 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# python train.py --name prob-ode/varx4-varv6 --exp toy --toy-exp spiral --reweight-type reciprocal --t-samp uniform --varx 4 
# --varv 6 --k 0.8 --sampling vanillaODE  --reweight reciprocal --probablistic-ode

# #Training
# python train.py --name train-toy/ode-spiral --exp toy --toy-exp spiral --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --ode --sampling vanillaODE --reweight reciprocal

# python train.py --name train-toy/ode-gmm    --exp toy --toy-exp gmm --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --ode --sampling vanillaODE --reweight reciprocal

# #Sampling 20 NFE
# python train.py --name eval-toy/ode-spiral --exp toy --toy-exp spiral --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --ode --sampling vanillaODE --reweight reciprocal --ckpt train-toy/ode-spiral --eval --interval 20

# python train.py --name eval-toy/ode-gmm    --exp toy --toy-exp gmm --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --ode --sampling vanillaODE --reweight reciprocal --ckpt train-toy/ode-gmm --eval --interval 20

# #Training
# python train.py --name train-toy/sde-spiral --exp toy --toy-exp spiral --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --sampling sscs --reweight reciprocal

# python train.py --name train-toy/sde-gmm    --exp toy --toy-exp gmm --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --sampling sscs --reweight reciprocal

# #Sampling 20 NFE
# python train.py --name eval-toy/sde-spiral --exp toy --toy-exp spiral --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --sampling sscs --reweight reciprocal --ckpt train-toy/sde-spiral --eval --interval 20

# python train.py --name eval-toy/sde-gmm    --exp toy --toy-exp gmm --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --sampling sscs --reweight reciprocal --ckpt train-toy/sde-gmm --eval --interval 20




#Source Command
# python train.py --name test-ode --exp toy --reweight-type ones --t-samp uniform --varx 1 --varv 10 --k 0.8 --ode --sampling vanillaODE --reweight reciprocal
