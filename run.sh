#! /usr/bin/env bash
# source ./home/ichuviliaeva/miniconda3/etc/profile.d/conda.sh
# conda init bash

torchrun --standalone --nnodes=1 --nproc_per_node=2 ./train.py "/DATA/ichuviliaeva/" --amp --aug-repeats=3 -b=16 --epochs=299 --channels-last --opt=lamb --weight-decay=0.02 --sched='cosine' --lr=5e-3 --warmup-epochs=5 --model=resnet50 --workers=2 --mixup=0.1 --cutmix=1.0 --amp --drop-path=0.05 --aa=rand-m7-mstd0.5-inc1 --reprob=0.0 --remode='pixel' --bce-loss --bce-target-thresh=0.2 --num-classes=16 --checkpoint-hist=10 --crop-pct=0.95 --seed=0 --smoothing=0.0 --train-interpolation='bicubic' --resume "./output/train/20220330-211934-resnet50-224/last.pth.tar"


#netstat -nltp to kill afterwardsnetstat -nltp to kill afterwards
