#! /usr/bin/env bash
# source ./home/ichuviliaeva/miniconda3/etc/profile.d/conda.sh
# conda init bashtorchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 ./train.py "./../data_rvl_cdip" --amp -b=16 --epochs=1 --channels-last --opt=lamb --weight-decay=0.02 --sched='cosine' --lr=5e-3 --warmup-epochs=1 --model=resnet50 --workers=4 --mixup=0.1 --cutmix=1.0 --amp --drop-path=0.05 --aa=rand-m7-mstd0.5-inc1 --reprob=0.0 --remode='pixel' --bce-loss --bce-target-thresh=0.2 --num-classes=16 --checkpoin
torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 ./train.py "/DATA/ichuviliaeva/" --amp -b=64 --epochs=1 --channels-last --opt=lamb --weight-decay=0.02 --sched='cosine' --lr=5e-3 --warmup-epochs=1 --model=resnet50 -j=8 --mixup=0.1 --cutmix=1.0 --amp --drop-path=0.05 --aa=rand-m7-mstd0.5-inc1 --reprob=0.0 --remode='pixel' --bce-loss --bce-target-thresh=0.2 --num-classes=16 --checkpoint-hist=10 --crop-pct=0.95 --seed=0 --smoothing=0.0 --train-interpolation='bicubic'



#netstat -nltp to kill afterwardsnetstat -nltp to kill afterwards
