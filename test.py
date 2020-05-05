import torch
import torch.nn.functional as F
import numpy as np


def test(args, model, device, test_loader_creator, logger):
    test_loaders_size = 0
    for test_loader in test_loader_creator.data_loaders:
        test_loaders_size += len(test_loader.dataset)

    model.eval()

    criterion = torch.nn.CrossEntropyLoss().to(device)

    test_loss = 0
    correct = 0
    label_correct = {}
    label_all = {}

    with torch.no_grad():
        for test_loader in test_loader_creator.data_loaders:
            for data, target in test_loader:

                data, target = data.to(device), target.to(device)
                
                output = model(data)
                
                test_loss += criterion(output, target).item()

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                for label in target.unique().tolist():
                    inds = (target == label)
                    corr = pred[inds,:].eq(target[inds].view_as(pred[inds,:])).sum().item()
                    if label not in label_correct:
                        label_correct[label] = 0
                        label_all[label] = 0
                    label_correct[label] += corr
                    label_all[label] += inds.sum().item()

    test_loss /= test_loaders_size

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, test_loaders_size,
        100. * correct / test_loaders_size))

    if args.acc_per_class:
        logger.info('Per class test accuracy:')
        per_class_acc = ''
        for label in sorted(label_all.keys()):
            per_class_acc += '{:4d}: {:4.0f}% '.format(label, 100. * label_correct[label]/label_all[label])
        logger.info(per_class_acc)
