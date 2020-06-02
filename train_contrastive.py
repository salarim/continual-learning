import copy
import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from termcolor import cprint

from log_utils import AverageMeter
from models.nearest_prototype import NearestPrototype
from optim import ContrastiveLoss
from test_contrastive import test_contrastive
from visualize import plot_embedding_tsne

def train_contrastive(args, model, device, train_loader_creator_l, train_loader_creator_u, 
                      test_loader_creator, logger):   
    nearest_proto_model = NearestPrototype(sigma=0.3)
    criterion =  ContrastiveLoss(device, args.batch_size, args.batch_size, 0.5, 0.5) # TODO
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loaders_l = train_loader_creator_l.data_loaders
    train_loaders_u = train_loader_creator_u.data_loaders
    for task_idx, (train_loader_l, train_loader_u) in enumerate(zip(train_loaders_l, train_loaders_u)):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader_l)\
                    * (args.epochs - args.cosine_annealing_warmup))
        
        old_model = copy.deepcopy(model)

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

                (_, output_l_1), (_, output_l_2) = model(data_l_1), model(data_l_2)
                (_, output_u_1), (_, output_u_2) = model(data_u_1), model(data_u_2)

                loss = criterion(output_l_1, output_l_2, output_u_1, output_u_2, target)

                loss.backward()                
                optimizer.step()

                losses.update(loss.item(), data_l_1.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {0} Epoch: [{1:3d}][{2:3d}/{3:3d}]\t'
                        'DTime {data_time.avg:.3f}\t'
                        'BTime {batch_time.avg:.3f}\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                            task_idx+1, epoch, batch_idx, len(train_loader_l),
                            batch_time=batch_time, data_time=data_time, loss=losses))
                
                if epoch >= args.cosine_annealing_warmup:
                    scheduler.step()

        for batch_idx, (data, _, target) in enumerate(train_loader_l):
            data, target = data.to(device), target.to(device)
            prev_feats = None
            if task_idx > 0:
                prev_feats, _ = old_model(data)
                prev_feats = prev_feats.detach()
            cur_feats, _ = model(data)
            cur_feats = cur_feats.detach()
            nearest_proto_model.add_features(task_idx, prev_feats, cur_feats, target)

        acc = AverageMeter()
        for batch_idx, (data, _, target) in enumerate(train_loader_l):
            data, target = data.to(device), target.to(device)
            cur_feats, _ = model(data)
            output = nearest_proto_model.predict(cur_feats)
            it_acc = (output == target).sum().item() / data.shape[0] 
            acc.update(it_acc, data.size(0))
        logger.info('Train task{:2d} Acc: {acc.avg:.3f}'.format((task_idx+1), acc=acc))

        test_contrastive(args, model, nearest_proto_model, device, test_loader_creator, logger)

        plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)
