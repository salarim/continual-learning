import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from termcolor import cprint

from log_utils import AverageMeter
from models.contrastive_wrapper import ProjectiveWrapper
from optim import ContrastiveLoss
# from visualize import plot_embedding_tsne

def train_contrastive(args, model, device, train_loader_creator_l, train_loader_creator_u, logger):   
    proj_model = ProjectiveWrapper(model, output_dim=64).to(device) # TODO
    criterion =  ContrastiveLoss(device, args.batch_size, args.batch_size, 0.07) # TODO
    optimizer = optim.SGD(proj_model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)

    train_loaders_l = train_loader_creator_l.data_loaders
    train_loaders_u = train_loader_creator_u.data_loaders
    for task_idx, (train_loader_l, train_loader_u) in enumerate(zip(train_loaders_l, train_loaders_u)):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        for epoch in range(1,args.epochs+1):
            
            model.train()
            losses = AverageMeter()

            for batch_idx, ((data_l_1, data_l_2, target), (data_u_1, data_u_2, _)) \
                in enumerate(zip(train_loader_l, train_loader_u)):

                data_l_1, data_l_2, target = data_l_1.to(device), data_l_2.to(device), target.to(device)
                data_u_1, data_u_2 = data_u_1.to(device), data_u_2.to(device)
                optimizer.zero_grad()

                output_l_1, output_l_2 = proj_model(data_l_1), proj_model(data_l_2)
                output_u_1, output_u_2 = proj_model(data_u_1), proj_model(data_u_2)

                loss = criterion(output_l_1, output_l_2, output_u_1, output_u_2, target)

                loss.backward()                
                optimizer.step()

                losses.update(loss.item(), data_l_1.size(0))

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {0} Epoch: [{1}][{2}/{3}]\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                            task_idx+1, epoch, batch_idx, len(train_loader_l),
                            loss=losses))

            scheduler.step()

        # plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            torch.save(model.state_dict(), model_path)
