#! /bin/bash

nohup sh -c "python3 -m tensorboard.main --logdir=/artifacts/log --port=25864" > /artifacts/log_tensorboard &
python3 train.py --dataset blender --dataset_folder abc3 --log_dir /artifacts/log --batch_size 32 --max_epoch 20 --dump_dir /artifacts/eval
