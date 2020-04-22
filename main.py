from __future__ import print_function
import argparse
import os
from time import localtime, strftime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from model import Net
from data_utils import DataloaderCreator
from train import train
from train_triplet import train_triplet
from test import test
from log_utils import makedirs, get_logger


def main():
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
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument("--save", type=str, default="experiments/")
    parser.add_argument('--model-type', type=str, default='softmax',
                        help='choose softmax or triplet')

    parser.add_argument('--dataset', type=str, default='mnist',
                        help='Name of dataset. (mnist/cifar100)')
    parser.add_argument('--tasks', type=int, default=2, metavar='N',
                    help='number of tasks (default: 2)')
    parser.add_argument('--exemplar-size', type=int, default=0, metavar='N',
                    help='number of exemplars (default: 0)')
    parser.add_argument('--oversample', action='store_true', default=False,
                        help='Oversample exemplars while training')
    parser.add_argument('--seprated-softmax', action='store_true', default=False,
                        help='Use seprated cross-entropy loss')
    args = parser.parse_args()

    makedirs(args.save)
    log_file = args.model_type + '-' + str(args.tasks) + '-'
    if args.oversample:
        log_file += 'OS-'
    if args.seprated_softmax:
        log_file += 'SS-'
    log_file += strftime("%Y-%m-%d-%H:%M:%S", localtime())
    python_files = [os.path.abspath(f) for f in os.listdir('.') \
        if os.path.isfile(f) and f.endswith('.py') and f != 'main.py']
    logger = get_logger(logpath=os.path.join(args.save, log_file),
     filepath=os.path.abspath(__file__),
     package_files=python_files)
    logger.info(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader_creator = DataloaderCreator(args, train=True, batch_size=args.batch_size,
        shuffle=False, **kwargs)

    test_loader_creator = DataloaderCreator(args, train=False, batch_size=args.test_batch_size,
        shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

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

    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()

