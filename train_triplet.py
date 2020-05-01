import torch
import torch.nn.functional as F
from termcolor import cprint

from test import test
from optim import triplet_loss
# from visualize import plot_embedding_tsne

def train_triplet(args, model, device, train_loader_creator, test_loader_creator, optimizer, logger):   
    T = 1
    model.train()
    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):
        buckets = train_loader_creator.buckets_list[task_idx]
        for epoch in range(1,args.epochs+1):
            for batch_idx, (data, target) in enumerate(train_loader):
                exemplar_data, exemplar_target = buckets[batch_idx]
                if exemplar_target is not None:
                    data = torch.cat((data, exemplar_data), 0)
                    target = torch.cat((target, exemplar_target), 0)
                data, target = data.to(device), target.to(device)
                anchor, pos, neg = data[:,0], data[:,1], data[:,2]
                optimizer.zero_grad()

                embedding_means = []
                for i in range(3):
                    embedding_list = []
                    for j in range(T):
                        embedding = model.get_embedding(data[:,i])
                        embedding_list.append(torch.unsqueeze(embedding, 0))
                    embedding_mean = torch.cat(embedding_list, 0).mean(0)
                    embedding_means.append(embedding_mean)

                anchor_emb, pos_emb, neg_emb = embedding_means[0], embedding_means[1], embedding_means[2]
                y = torch.FloatTensor(anchor_emb.shape[0]).fill_(-1).to(device)
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb, y)
                loss.backward()                
                optimizer.step()

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {} Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        task_idx+1, epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                        100. * (batch_idx * args.batch_size) / len(train_loader.dataset), loss.item()))
        
        # plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            torch.save(model.state_dict(), model_path)
