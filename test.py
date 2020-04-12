import torch
import torch.nn.functional as F
import numpy as np


def test(args, model, device, test_loader_creator, logger, print_entropy=True):
    test_loaders_size = 0
    for test_loader in test_loader_creator.data_loaders:
        test_loaders_size += len(test_loader.dataset)

    model.eval()
    T = 10
    test_loss = 0
    correct = 0
    label_correct = {}
    label_all = {}
    output_variances = {i:[] for i in range(10)}
    output_entropies = {i:[] for i in range(10)}

    with torch.no_grad():
        for test_loader in test_loader_creator.data_loaders:
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output_list = []
                for i in range(T):
                    output_list.append(torch.unsqueeze(model(data), 0))
                output_mean = torch.cat(output_list, 0).mean(0)
                output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
                output_entropy = (-output_mean.exp() * output_mean).sum(dim=1).mean().item()
                test_loss += F.nll_loss(output_mean, target, reduction='sum').item()
                pred = output_mean.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

                for label in range(10):
                    inds = (target == label)
                    corr = pred[inds,:].eq(target[inds].view_as(pred[inds,:])).sum().item()
                    if label not in label_correct:
                        label_correct[label] = 0
                        label_all[label] = 0
                    label_correct[label] += corr
                    label_all[label] += inds.sum().item()

                for label in target.unique().tolist():
                    output_variances[label].append(output_variance)
                    output_entropies[label].append(output_entropy)

    test_loss /= test_loaders_size

    logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, test_loaders_size,
        100. * correct / test_loaders_size))

    logger.info('Per class test accuracy:')
    per_class_acc = ''
    for label in range(10):
        per_class_acc += '{:4d}: {:4.0f}% '.format(label, 100. * label_correct[label]/label_all[label])
    logger.info(per_class_acc)
    
    if print_entropy:
        logger.info('Class Variance/Entropy:')
        for label in range(10):
            logger.info(str(label) + ' ' + \
                str(np.mean(output_variances[label])) + ' ' + \
                str(np.mean(output_entropies[label])))
        logger.info('\n')
