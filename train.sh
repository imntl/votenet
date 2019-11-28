#! /bin/bash

#nohup sh -c "python3 -m tensorboard.main --logdir=/artifacts/log --port=25864" > /artifacts/log_tensorboard &
CUDA_VISIBLE_DEVICES=1 python3 train.py --dataset blender --dataset_folder abc6_small --log_dir /artifacts/log --batch_size 13 --max_epoch 200 --dump_dir /artifacts/eval
