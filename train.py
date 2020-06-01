import time
from termcolor import cprint

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR

from test import test, accuracy
from log_utils import AverageMeter
# from visualize import plot_embedding_tsne

def train(args, model, device, train_loader_creator, test_loader_creator, logger):   

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)

    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        for epoch in range(1,args.epochs+1):
            
            model.train()
            losses = AverageMeter()
            acc = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            end = time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                data_time.update(time.time() - end)

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                _, output = model(data)

                loss = criterion(output, target)

                loss.backward()                
                optimizer.step()

                it_acc = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                acc.update(it_acc.item(), data.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {0} Epoch: [{1:3d}][{2:3d}/{3:3d}]\t'
                        'DTime {data_time.avg:.3f}\t'
                        'BTime {batch_time.avg:.3f}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            task_idx+1, epoch, batch_idx, len(train_loader),
                            batch_time=batch_time, data_time=data_time, loss=losses, acc=acc))

            scheduler.step()
            if epoch % args.test_interval == 0:
                test(args, model, device, test_loader_creator, logger)

        # plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
