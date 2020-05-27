import torch

from log_utils import AverageMeter

def test_contrastive(args, model, nearest_proto_model, device, test_loader_creator_l, logger): 
    model.eval()  

    acc = AverageMeter()
    tasks_acc = [AverageMeter() for i in range(len(test_loader_creator_l.data_loaders))]

    test_loaders_l = test_loader_creator_l.data_loaders

    with torch.no_grad():
        for task_idx, test_loader_l in enumerate(test_loaders_l):

            for batch_idx, (data, _, target) in enumerate(test_loader_l):
                data, target = data.to(device), target.to(device)
                cur_feats = model.get_embedding(data)
                output = nearest_proto_model.predict(cur_feats)
                it_acc = (output == target).sum().item() / data.shape[0] 
                acc.update(it_acc, data.size(0))
                tasks_acc[task_idx].update(it_acc, data.size(0))
    
    if args.acc_per_task:
        tasks_acc_str = 'Tess Acc per task: '
        for i, task_acc in enumerate(tasks_acc):
            tasks_acc_str += 'Task{:2d} Acc: {acc.avg:.3f}'.format((i+1), acc=task_acc) + '\t'
        logger.info(tasks_acc_str)
    logger.info('Test Acc: {acc.avg:.3f}'.format(acc=acc))
