import copy
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from termcolor import cprint

from log_utils import AverageMeter
from models.nearest_prototypes import NearestPrototypes
from models.nearest_prototype import NearestPrototype
from models.nearest_stream_prototype import NearestStreamPrototype
from optim import ContrastiveLoss, warmup_learning_rate
from test_contrastive import test_contrastive
from test import accuracy
from visualize import plot_embedding_tsne

def train_contrastive(args, model, device, train_loader_creator_l, train_loader_creator_u, 
                      test_loader_creator, logger):   
    nearest_proto_model = NearestStreamPrototype(sigma=args.sigma)
    criterion =  ContrastiveLoss(device, args.batch_size, args.batch_size, args.temp, args.sup_coef)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    train_loaders_l = train_loader_creator_l.data_loaders
    train_loaders_u = train_loader_creator_u.data_loaders
    for task_idx, (train_loader_l, train_loader_u) in enumerate(zip(train_loaders_l, train_loaders_u)):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                               eta_min=args.lr * (args.gamma ** 3), 
                                                               T_max=max(args.epochs,1))

        for epoch in range(1,args.epochs+1):
            
            model.train()
            losses = AverageMeter()
            batch_time = AverageMeter()
            data_time = AverageMeter()

            end = time.time()
            for batch_idx, ((data_l_1, data_l_2, target), (data_u_1, data_u_2, _)) \
                in enumerate(zip(train_loader_l, train_loader_u)):
                data_time.update(time.time() - end)

                data_l_1, data_l_2, target = data_l_1.to(device), data_l_2.to(device), target.to(device)
                data_u_1, data_u_2 = data_u_1.to(device), data_u_2.to(device)
                optimizer.zero_grad()

                (hidden_l_1, output_l_1), (_, output_l_2) = model(data_l_1), model(data_l_2)
                (_, output_u_1), (_, output_u_2) = model(data_u_1), model(data_u_2)

                loss = criterion(output_l_1, output_l_2, output_u_1, output_u_2, target)
                loss.backward()

                warmup_learning_rate(args, epoch, batch_idx, len(train_loader_l), optimizer)               
                optimizer.step()

                losses.update(loss.item(), data_l_1.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {0} Epoch: [{1:3d}][{2:3d}/{3:3d}]\t'
                        'DTime {data_time.avg:.3f}\t'
                        'BTime {batch_time.avg:.3f}\t'
                        'LR {lr:.3f}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                            task_idx+1, epoch, batch_idx, len(train_loader_l),
                            batch_time=batch_time, data_time=data_time, 
                            lr=optimizer.param_groups[0]['lr'], loss=losses))

                new_hidden_l_1, _ = model(data_l_1)
                nearest_proto_model.add_features(hidden_l_1.detach(), new_hidden_l_1.detach(), target)
                            
            scheduler.step()

        model.eval()

        acc = AverageMeter()
        for batch_idx, (data, _, target) in enumerate(train_loader_l):
            data, target = data.to(device), target.to(device)
            cur_feats, _ = model(data)
            output = nearest_proto_model.predict(cur_feats)
            it_acc = (output == target).sum().item() / data.shape[0] 
            acc.update(it_acc, data.size(0))
        logger.info('Train task{:2d} Acc: {acc.avg:.3f}'.format((task_idx+1), acc=acc))

        test_contrastive(args, model, nearest_proto_model, device, test_loader_creator, logger)
        test_features(args, model, device, train_loader_creator_l, test_loader_creator, logger)

        plot_embedding_tsne(args, task_idx, test_loader_creator, model, device, nearest_proto_model)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

def test_features(args, model, device, train_loader_creator_l, test_loader_creator, logger):
    logger.info('Evaluating linear model Began.')
    if isinstance(model, torch.nn.DataParallel):
        num_ftrs = model.module.model.classifier.in_features
        num_out = model.module.model.classifier.out_features
    else:
        num_ftrs = model.model.classifier.in_features
        num_out = model.model.classifier.out_features
    linear_model = torch.nn.Linear(num_ftrs, num_out).to(device)

    lr = 0.1
    epochs = 10
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(linear_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    for task_idx, train_loader in enumerate(train_loader_creator_l.data_loaders):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        for epoch in range(1,epochs+1):
            
            linear_model.train()
            losses = AverageMeter()
            acc = AverageMeter()

            for batch_idx, (data, _, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                h = model(data)[0].detach()
                output = linear_model(h)

                loss = criterion(output, target)

                loss.backward()                
                optimizer.step()

                it_acc = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                acc.update(it_acc.item(), data.size(0))

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {0} Epoch: [{1:3d}][{2:3d}/{3:3d}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(
                            task_idx+1, epoch, batch_idx, len(train_loader), loss=losses, acc=acc))

            scheduler.step()

    linear_model.eval()

    with torch.no_grad():
        losses = AverageMeter()
        acc = AverageMeter()
        
        for test_loader in test_loader_creator.data_loaders:

            for data, _, target in test_loader:

                data, target = data.to(device), target.to(device)
                h = model(data)[0].detach()
                output = linear_model(h)

                loss = criterion(output, target)

                output = output.float()
                loss = loss.float()

                it_acc = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                acc.update(it_acc.item(), data.size(0))

    logger.info('Test set: Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.avg:.3f}'.format(
                loss=losses, acc=acc))
    logger.info('Evaluating linear model Finished.')
