from __future__ import print_function
import argparse
import os
from time import localtime, strftime
import warnings

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch.nn.parallel
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from model import get_model
from data_utils import DataLoaderConstructor
from train import train
from train_triplet import train_triplet
from test import test
from log_utils import makedirs, get_logger

parser = argparse.ArgumentParser(description='PyTorch Longlife Learning')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=5, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--seed', default=None, type=int,
                help='seed for initializing training. ')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
parser.add_argument("--save", type=str, default="experiments/")
parser.add_argument('--model-type', type=str, default='softmax',
                    help='choose softmax or triplet')

parser.add_argument('--dataset', type=str, default='mnist',
                    help='Name of dataset. (mnist/cifar10/cifar100/imagenet)')
parser.add_argument('--tasks', type=int, default=2, metavar='N',
                help='number of tasks (default: 2)')
parser.add_argument('--exemplar-size', type=int, default=0, metavar='N',
                help='number of exemplars (default: 0)')
parser.add_argument('--oversample-ratio', type=float, default=0.0, metavar='M',
                help='oversampling ratio (default: 0.0')
parser.add_argument('--seprated-softmax', action='store_true', default=False,
                    help='Use seprated cross-entropy loss')

# device arguments
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')
parser.add_argument('--gpu', default=None, type=int,
                help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                help='distributed backend')
parser.add_argument('--world-size', default=-1, type=int,
                help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                help='node rank for distributed training')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                help='Use multi-processing distributed training to launch '
                        'N processes per node, which has N GPUs. This is the '
                        'fastest way to use PyTorch for either single node or '
                        'multi node data parallel training')


def main():
    args = parser.parse_args()

    makedirs(args.save)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    log_file = args.model_type + '-' + str(args.tasks) + '-'
    if args.oversample_ratio > 0.0:
        log_file += 'OS-'
    if args.seprated_softmax:
        log_file += 'SS-'
    log_file += strftime("%Y-%m-%d-%H#%M#%S", localtime()) + '-'
    log_file += str(gpu)
    python_files = [os.path.abspath(f) for f in os.listdir('.') \
        if os.path.isfile(f) and f.endswith('.py') and f != 'main.py']
    logger = get_logger(logpath=os.path.join(args.save, log_file),
     filepath=os.path.abspath(__file__),
     package_files=python_files)
    logger.info(args)

    args.gpu = gpu
    if args.gpu is not None:
        logger.info("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    model = get_model(args)
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.test_batch_size = int(args.test_batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    # else:
    #     # DataParallel will divide and allocate batch_size to all available GPUs
    #     model = torch.nn.DataParallel(model).cuda() #TODO I'm not sure about it. just copied.



    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.gpu else {}
    train_loader_creator = DataLoaderConstructor(args, train=True, **kwargs)

    test_loader_creator = DataLoaderConstructor(args, train=False, **kwargs)

    device = torch.device("cuda:{}".format(args.gpu) if args.gpu is not None else "cpu")
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)

    if args.save_model:
        torch.save(model.state_dict(), "initial.pt")

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    args.vis_base_dir = 'plots/' + log_file + '/'
    if args.model_type == 'softmax':
        train(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger)
        test(args, model, device, test_loader_creator, logger)
    elif args.model_type == 'triplet':
        train_triplet(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger)
    # scheduler.step()



if __name__ == '__main__':
    main()
