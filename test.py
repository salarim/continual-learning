import torch
import torch.nn.functional as F
import numpy as np

from log_utils import AverageMeter

def test(args, model, device, test_loader_creator, logger):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        losses = AverageMeter()
        acc = AverageMeter()
        
        for test_loader in test_loader_creator.data_loaders:

            for data, target in test_loader:

                data, target = data.to(device), target.to(device)
                _, output = model(data)

                loss = criterion(output, target)

                output = output.float()
                loss = loss.float()

                it_acc = accuracy(output.data, target)[0]
                losses.update(loss.item(), data.size(0))
                acc.update(it_acc.item(), data.size(0))

    logger.info('Test set: Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Acc {acc.avg:.3f}'.format(
                loss=losses, acc=acc))

    # TODO calculate accuracy per class.

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
