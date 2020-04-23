import torch
import numpy as np
from tsnecuda import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from log_utils import makedirs


def plot_embedding_tsne(args, task_id, data_loader_creator, model, device):
    embedding_size = model.fc.in_features
    X = np.empty((0,embedding_size), dtype=np.float32)
    targets = np.empty((0))
    with torch.no_grad():
        for data_loader in data_loader_creator.data_loaders:
            for data, target in data_loader:
                data = data.to(device)
                if args.model_type == 'triplet':
                    data = data[:,0]
                embedding = model.get_embedding(data)
                embedding = embedding.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                X = np.append(X, embedding, axis=0)
                targets = np.append(targets, target)
    
    X_tsne = TSNE().fit_transform(X)

    dir_name = args.vis_base_dir + 'T' + str(task_id+1) + '/'
    makedirs(dir_name)

    for t_id, task in enumerate(data_loader_creator.task_target_set):
        plt.figure()
        palette = sns.color_palette("bright", len(task))
        idx = np.isin(targets, task)
        sns_plot = sns.scatterplot(X_tsne[idx,0], X_tsne[idx,1], hue=targets[idx], legend='full', palette=palette, s=20)
        plt.savefig(dir_name + 'T' + str(t_id+1) + '.png')

    plt.figure()
    palette = sns.color_palette("bright", np.unique(targets).shape[0])
    sns_plot = sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=targets, legend='full', palette=palette, s=20)
    plt.savefig(dir_name + 'all.png')
    print('Visualization Ended.')

