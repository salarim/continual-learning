from __future__ import print_function
import argparse
import os
import random
from time import localtime, strftime
import warnings

import torch
import torch.backends.cudnn as cudnn

from model import get_model
from data_utils import DataConfig, DataLoaderConstructor
from train import train
from train_triplet import train_triplet
from train_contrastive import train_contrastive
from test import test
from log_utils import makedirs, get_logger

parser = argparse.ArgumentParser(description='PyTorch Longlife Learning')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.2, metavar='M',
                    help='Learning rate step gamma (default: 0.2)')
parser.add_argument('--milestones', type=int, default=[60, 120, 160], nargs='+')
parser.add_argument('--warm-epochs', type=int, default=0, metavar='N',
                    help='number of epochs for warmup(default: 0)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    help='Optimizer weight dacay (default: 5e-4)')
parser.add_argument('--seed', default=None, type=int,
                help='seed for initializing training. ')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before testing model')
parser.add_argument('--acc-per-task', action='store_true', default=False,
                    help='log accuracy of model per class')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument("--save", type=str, default="experiments/")
parser.add_argument('--model-type', type=str, default='softmax',
                    help='choose softmax/triplet/contrastive')
parser.add_argument('--model-path', type=str, default='',
                    help='The path that trained model located.')

parser.add_argument('--dataset', type=str, default='mnist',
                    help='Name of dataset. (mnist/cifar10/cifar100/imagenet) (default: mnist')
parser.add_argument('--unlabeled-dataset', type=str, default='mnist',
                    help='Name of unlabeled dataset. (mnist/cifar10/cifar100/imagenet) (default: mnist)')
parser.add_argument('--data-path', type=str, 
                    default='/local-scratch2/hadi/datasets/ILSVRC/Data/CLS-LOC',
                    help='Path of data (default: my own computer path!')
parser.add_argument('--tasks', type=int, default=2, metavar='N',
                help='number of tasks (default: 2)')
parser.add_argument('--exemplar-size', type=int, default=0, metavar='N',
                help='number of exemplars (default: 0)')
parser.add_argument('--oversample-ratio', type=float, default=0.0, metavar='M',
                help='oversampling ratio (default: 0.0')

# device arguments
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                help='GPU id to use.')

parser.add_argument('--sup-coef', type=float, default=0.5, metavar='LR',
                    help='Supervised coefficient in ContrastiveLoss. (default: 0.5)')
parser.add_argument('--temp', type=float, default=0.5,
                    help='temperature for loss function')
parser.add_argument('--proj-dim', type=int, default=256, metavar='N',
                help='Dimension of mlp projection output (default: 256)')
parser.add_argument('--sigma', type=float, default=0.01,
                    help='Sigma for the NearestPrototype model (default: 0.01)')


def main():
    args = parser.parse_args()

    makedirs(args.save)

    log_file = args.model_type + '-' + str(args.tasks) + '-'
    if args.oversample_ratio > 0.0:
        log_file += 'OS-'
    log_file += strftime("%Y-%m-%d-%H#%M#%S", localtime()) + '-'
    python_files = [os.path.abspath(f) for f in os.listdir('.') \
        if os.path.isfile(f) and f.endswith('.py') and f != 'main.py']
    logger = get_logger(logpath=os.path.join(args.save, log_file),
     filepath=os.path.abspath(__file__),
     package_files=python_files)

    logger.info(args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    model = get_model(args)

    device = torch.device("cuda:{}".format(args.gpu) if args.gpu is not None else "cpu")
    if args.gpu is not None:
        logger.info("Use GPU {} for training".format(args.gpu))
        model = model.cuda()
        device = torch.device("cuda:{}".format(args.gpu))
    elif torch.cuda.device_count() > 0:
        logger.info("Use {} GPU/GPUs for training".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model).cuda()
        device = torch.device("cuda")
    else:
        logger.info("Use CPU for training")
        device = torch.device("cpu")


    train_loader_creator_config = DataConfig(args, train=True, dataset=args.dataset,
                                             dataset_type=args.model_type, is_continual=True, 
                                             batch_size=args.batch_size)
    train_loader_creator = DataLoaderConstructor(train_loader_creator_config)

    if args.model_type == 'contrastive':
        train_loader_creator_u_config = DataConfig(args, train=True, dataset=args.unlabeled_dataset,
                                                   dataset_type=args.model_type, is_continual=False, 
                                                   batch_size=args.batch_size)
        train_loader_creator_u = DataLoaderConstructor(train_loader_creator_u_config)

    test_loader_creator_config = DataConfig(args, train=False, dataset=args.dataset,
                                            dataset_type=args.model_type, is_continual=True, 
                                            batch_size=args.test_batch_size, exemplar_size=0)
    test_loader_creator = DataLoaderConstructor(test_loader_creator_config)

    args.vis_base_dir = 'plots/' + log_file + '/'
    if args.model_type == 'softmax':
        train(args, model, device, train_loader_creator,
              test_loader_creator, logger)
        test(args, model, device, test_loader_creator, logger)
    elif args.model_type == 'triplet':
        train_triplet(args, model, device, train_loader_creator,
                      test_loader_creator, logger)
    elif args.model_type == 'contrastive':
        train_contrastive(args, model, device, train_loader_creator,
                          train_loader_creator_u, test_loader_creator, logger)



if __name__ == '__main__':
    main()
