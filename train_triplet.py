import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from termcolor import cprint

from test import test
from optim import triplet_loss
# from visualize import plot_embedding_tsne

def train_triplet(args, model, device, train_loader_creator, test_loader_creator, logger):   
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=0.9, weight_decay=args.weight_decay)

    for task_idx, train_loader in enumerate(train_loader_creator.data_loaders):

        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr
        scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

        for epoch in range(1,args.epochs+1):
            for batch_idx, (data, target) in enumerate(train_loader):

                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()

                embeddings = []
                for i in range(3):
                    embedding = model.get_embedding(data[:,i])
                    embeddings.append(embedding)
                anchor_emb, pos_emb, neg_emb = embeddings[0], embeddings[1], embeddings[2]

                y = torch.FloatTensor(anchor_emb.shape[0]).fill_(-1).to(device)
                loss = triplet_loss(anchor_emb, pos_emb, neg_emb, y)
                
                loss.backward()                
                optimizer.step()
                scheduler.step()

                if batch_idx % args.log_interval == 0:
                    logger.info('Train Task: {} Epoch: {} '
                                '[{:7d}/{:7d} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                                task_idx+1, epoch, batch_idx * args.batch_size,
                                len(train_loader.dataset),
                                100. * (batch_idx * args.batch_size) / len(train_loader.dataset),
                                loss.item()))
        
        # plot_embedding_tsne(args, task_idx, test_loader_creator, model, device)
        if args.save_model:
            model_path = args.vis_base_dir.split('/')[-2] + 'T' + str(task_idx+1) + '.pt'
            torch.save(model.state_dict(), model_path)
