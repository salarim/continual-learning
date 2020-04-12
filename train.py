import torch
import torch.nn.functional as F
from termcolor import cprint

from test import test
from optim import seprated_softmax_loss

def train(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger):   
    T = 10
    model.train()
    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):
        if task_idx > 0:
            buckets = train_loader_creator.buckets_list[task_idx-1]
        for epoch in range(1,args.epochs+1):
            for batch_idx, (data, target) in enumerate(train_loader):
                if task_idx > 0:
                    exemplar_data, exemplar_target = buckets[batch_idx]
                    if exemplar_target is not None:
                        data = torch.cat((data, exemplar_data), 0)
                        target = torch.cat((target, exemplar_target), 0)
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                output_list = []
                score_list = []
                for i in range(T):
                    score, output = model(data)
                    output_list.append(torch.unsqueeze(output, 0))
                    score_list.append(torch.unsqueeze(score, 0))
                output_mean = torch.cat(output_list, 0).mean(0)
                score_mean = torch.cat(score_list, 0).mean(0)
                output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
                output_entropy = (-output_mean.exp() * output_mean).sum(dim=1).mean().item()
                # loss = F.nll_loss(output_mean, target)
                loss = seprated_softmax_loss(score_mean, target, train_loader_creator.task_target_set, task_idx)
                loss.backward()
                # Change lr
                # scaled_entropy = output_entropy * 100.
                # new_lr = args.lr / min(max(scaled_entropy, 1.0), 100.0)
                # logger.info('New Learning Rate: {:.5f}'.format(new_lr))
                # for param_group in optimizer.param_groups:
                #         param_group['lr'] = new_lr
                
                # if min(target).item() < min(train_loader_creator.task_target_set[task_idx]):
                #     new_lr = args.lr
                # else:
                #     new_lr = args.lr / 50
                # logger.info('New Learning Rate: {:.5f}'.format(new_lr))
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = new_lr
                
                optimizer.step()

                pred = output_mean.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()

                if batch_idx % args.log_interval == 0:
                    logger.info('Batch labels: ' + str(torch.unique(target).tolist()))
                    logger.info('Train Task: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Batch_Acc: {:.2f} Entropy: {:.6f} Variance: {:.6f}'.format(
                        task_idx+1, epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                        100. * (batch_idx * args.batch_size) / len(train_loader.dataset), loss.item(), correct / target.shape[0],
                        output_entropy, output_variance))

            test(args, model, device, test_loader_creator, logger)
