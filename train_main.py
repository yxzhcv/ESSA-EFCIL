import torch
import numpy as np
import argparse

import random
import os
import logging
import sys
import time
# import yaml

from utils.Trainer_essa import Trainer


def main():
    parser = argparse.ArgumentParser(description='my_IL')

    # dir
    parser.add_argument('--data_dir', type=str, default='./dataset', help='dataset dir (default: ./dataset)')

    # dataset option
    parser.add_argument('--dataset_name', type=str, default='CIFAR100',
                        help='dataset name (default: my)')                         
    parser.add_argument('--dataset_type', type=str, default='cl',
                        help='type of preprocessing dataset (default: my)')
    parser.add_argument('--loader_type', type=str, default='normal',
                        help='type of preprocessing dataloader (default: normal)')

    # dataset setting(class-division, way, shot)
    parser.add_argument('--base_class', type=int, default=50, help='number of base class (default: 60)')
    parser.add_argument('--way', type=int, default=5, help='class number of per task (default: 5)')
    parser.add_argument('--shot', type=int, default=5, help='shot of per class (default: 5)')
    parser.add_argument('--orders', type=list, default=None, help='order of datasets class (default: None)')
    parser.add_argument('--orders_path', type=str, default='cifar100_orders.yaml',
                        help='path of order file (default: None)')
    parser.add_argument('--no_order', action='store_true', default=False, help='no order during training')

    # model option
    parser.add_argument('--model_name', type=str, default='my', help='model name (default: my)')
    parser.add_argument('--model_path', type=str, default='./pre/my/model_CIFAR100_pre50.pth', help='model path (default: None)')
    parser.add_argument('--pretrained', action='store_true', default=False, help='pretrained model')
    parser.add_argument('--backbone_name', type=str, default='resnet18_dsr', help='backbone name (default: resnet18)')
    parser.add_argument('--init_fic', type=str, default='None', choices=['None', 'identical'],
                        help='init_fic name (default: None)')
    parser.add_argument('--batch_task', type=int, default=3, help='tasks per batch')
    parser.add_argument('--embedding', type=int, default=64, choices=[64, 128, 256, 512], help='channel of embedding')
    parser.add_argument('--latent_dim', type=int, default=512, help='channel of latent')
    parser.add_argument('--classifier', type=str, default='fc_IL_base', help='classifier name (default: nce)')

    # backbone
    parser.add_argument('--mode', type=str, default='normal', choices=['parallel_adapters', 'normal', 'film'],
                        help='type of adapters (default: normal)')
    parser.add_argument('--base_train', type=str, default='base', help='base_train model')

    # gpu option
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--cudnn', action='store_true', default=True, help='enables CUDNN accelerate')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                            comma-separated list of integers only (default=0)')

    # loss option
    parser.add_argument('--loss_type', type=str, default='ce', help='type of loss (default: ce)')
    parser.add_argument('--loss_weight', type=int, default=1, metavar='N', help='weight ratio of loss (default: 1)')

    # random seed
    parser.add_argument('--seed', type=int, default=1993, help='random seed (default: 1993)')

    # hyper option
    parser.add_argument('--session', type=int, default=11, metavar='N', help='training session (default:6)')
    parser.add_argument('--base_epochs', type=int, default=100, metavar='N', help='base epochs (default:60)')
    parser.add_argument('--new_epochs', type=int, default=60, metavar='N', help='base epochs (default:60)')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch_size (default:128)')
    parser.add_argument('--batch_size_test', type=int, default=128, metavar='N', help='test batch_size (default:128)')

    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--lr_new', type=float, default=0.0002, metavar='LR', help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='step',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--no-lr-scheduler', '-a', action='store_true', default=False,
                        help='avoid lr-schduler (default: False)')
    parser.add_argument('--lr_coefficient', nargs='+', type=float, default=[1, 1, 1, 1], help='list of lr_coefficient')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4, metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether use nesterov (default: False)')

    # optimizer option
    parser.add_argument('--opt', type=str, default='opt1', choices=['opt1', 'opt2', 'opt3'],
                        help='type of learnable para (default: opt1)')
    parser.add_argument('--optim', type=str, default='adam', choices=['sgd', 'adam'],
                        help='type of optimizer (default: sgd)')

    # evaluation options
    parser.add_argument('--eval-interval', type=int, default=1, help='evaluation interval (default: 1)')
    parser.add_argument('--val', action='store_true', default=False, help='val mode')

    # saver
    parser.add_argument('--checkname', type=str, default='my-IL', help='set the checkpoint name')
    
    # Hyper-parameter
    parser.add_argument('--lam', type=float, default=3, help='lambda (default: 1)')
    parser.add_argument('--alp', type=float, default=100, help='alpha (default: 100)')
    parser.add_argument('--beta', type=float, default=10, help='beta (default: 10)')

    args = parser.parse_args()

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.data_path = os.path.join(args.data_dir, args.dataset_name)
    if args.cuda:
        try:
            args.gpu_list = [int(ids) for ids in args.gpu_ids.split(',')]
            args.gpu_num = len(args.gpu_list)
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    args.tasks = args.session - 1
    args.all_class = args.base_class+args.way*args.tasks

    dir_log = './log/essa_log/'
    if not os.path.exists(dir_log):
        os.mkdir(dir_log)
    args.dir_name = dir_log + str(args.dataset_name) + '_' + str(args.model_name) + '_' + str(args.init_fic) \
                    + '_' + str(args.lr)
    if not os.path.exists(args.dir_name):
        os.mkdir(args.dir_name)
    args.now_time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    logging.basicConfig(level=logging.INFO,
                        filename=args.dir_name + '/output_' + args.now_time + '.log',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if args.cudnn:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    trainer = Trainer(args)
    for session in range(args.session):
        print('****** Task: %d ******' % session)
        trainer.training(session)
        trainer.validation(session, print=True)


if __name__ == "__main__":
    main()
