import torch
import torch.nn.functional as F
from termcolor import cprint

from test import test
from optim import seprated_softmax_loss
from visualize import plot_embedding_tsne

def train(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger):   
    T = 1
    model.train()
    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):
        buckets = train_loader_creator.buckets_list[task_idx]
        for epoch in range(1,args.epochs+1):
            target_size = {}
            for batch_idx, (data, target) in enumerate(train_loader):
                exemplar_data, exemplar_target = buckets[batch_idx]
                if exemplar_target is not None:
                    data = torch.cat((data, exemplar_data), 0)
                    target = torch.cat((target, exemplar_target), 0)
                for i in target.unique().tolist():
                    if i not in target_size:
                        target_size[i] = 0
                    target_size[i] += torch.sum(target == i).item()
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
                if args.seprated_softmax:
                    loss = seprated_softmax_loss(score_mean, target,
                     train_loader_creator.task_target_set, task_idx)
                else:
                    loss = F.nll_loss(output_mean, target)
                loss.backward()                
                optimizer.step()

                pred = output_mean.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()

                if batch_idx % args.log_interval == 0:
                    # logger.info('Batch labels: ' + str(target.unique().tolist()))
                    logger.info('Train Task: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Batch_Acc: {:.2f} Entropy: {:.6f} Variance: {:.6f}'.format(
                        task_idx+1, epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                        100. * (batch_idx * args.batch_size) / len(train_loader.dataset), loss.item(), correct / target.shape[0],
                        output_entropy, output_variance))

            # logger.info('Targets size this epoch:' + ' '.join([str(k) + ':' + str(v) + ',' for k,v in target_size.items()]))
            test(args, model, device, test_loader_creator, logger)

        plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            torch.save(model.state_dict(), model_path)
