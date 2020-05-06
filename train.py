import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from termcolor import cprint

from test import test, accuracy
from log_utils import AverageMeter
# from visualize import plot_embedding_tsne

def train(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger):   
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=args.gamma)

        for epoch in range(1,args.epochs+1):

            losses = AverageMeter()
            acc = AverageMeter()

            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, target)

                loss.backward()                
                optimizer.step()
                scheduler.step()

                it_acc = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                acc.update(it_acc.item(), data.size(0))

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {0} Epoch: [{1}][{2}/{3}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            task_idx+1, epoch, batch_idx, len(train_loader),
                            loss=losses, acc=acc))

            if epoch % args.test_interval == 0:
                test(args, model, device, test_loader_creator, logger)

        # plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            torch.save(model.state_dict(), model_path)
