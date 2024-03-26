#!/bin/sh

python train_main.py --backbone_name resnet18_dsr --model_name my --dataset_type cl --optim adam --opt opt1 \
--loss_type ce --no_order --seed 1993 --base_class 50 --way 5 --session 11 --lr 0.001 --lr-scheduler step \
--batch_size 128 --gpu-ids 0 --base_epochs 60 --new_epochs 60 --mode 'normal' --weight-decay 5e-4 \
--data_dir './dataset' --batch_size_test 128 --classifier fc_IL_base  \
--val --model_path './pre/my/model_CIFAR100_pre50.pth'

