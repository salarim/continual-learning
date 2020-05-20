import torch

from log_utils import AverageMeter

def test_contrastive(args, model, nearest_proto_model, device, test_loader_creator_l, logger): 
    model.eval()  

    acc = AverageMeter()

    test_loaders_l = test_loader_creator_l.data_loaders

    with torch.no_grad():
        for test_loader_l in test_loaders_l:

            for batch_idx, (data, _, target) in enumerate(test_loader_l):
                data, target = data.to(device), target.to(device)
                cur_feats = model.get_embedding(data)
                output = nearest_proto_model.predict(cur_feats)
                it_acc = (output == target).sum().item() / data.shape[0] 
                acc.update(it_acc, data.size(0))
    
    print('Test Acc: {acc.avg:.3f}'.format(acc=acc))
