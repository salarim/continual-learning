import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from termcolor import cprint

from test import test
# from visualize import plot_embedding_tsne

def train(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger):   
    model.train()
    
    criterion = torch.nn.CrossEntropyLoss().to(device)

    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = MultiStepLR(optimizer, milestones=[100, 150], gamma=args.gamma)

        for epoch in range(1,args.epochs+1):
            target_size = {}
            for batch_idx, (data, target) in enumerate(train_loader):

                for i in target.unique().tolist():
                    if i not in target_size:
                        target_size[i] = 0
                    target_size[i] += torch.sum(target == i).item()

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output = model(data)

                loss = criterion(output, target)

                loss.backward()                
                optimizer.step()
                scheduler.step()

                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()

                if batch_idx % args.log_interval == 0:
                    # logger.info('Batch labels: ' + str(target.unique().tolist()))
                    logger.info('Train Task: {} Epoch: {}'
                                '[{:7d}/{:7d} ({:3.0f}%)]'
                                '\tLoss: {:.6f} Batch_Acc: {:.2f}'.format(
                                task_idx+1, epoch,
                                batch_idx * args.batch_size, len(train_loader.dataset),
                                100. * (batch_idx * args.batch_size) / len(train_loader.dataset),
                                loss.item(), correct / target.shape[0]))

            if epoch % args.test_interval == 0:
                test(args, model, device, test_loader_creator, logger)

        # plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            torch.save(model.state_dict(), model_path)
