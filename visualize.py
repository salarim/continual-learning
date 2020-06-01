import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from log_utils import makedirs

try:
    from tsnecuda import TSNE
except ImportError:
    from sklearn.manifold import TSNE


def plot_embedding_tsne(args, task_id, data_loader_creator, model, device):
    X = None
    targets = np.empty((0))
    with torch.no_grad():
        for data_loader in data_loader_creator.data_loaders:
            for data_target in data_loader:
                if data_loader_creator.config.dataset_type in ['softmax', 'triplet']:
                    data, target = data_target
                elif data_loader_creator.config.dataset_type == 'contrastive':
                    data, _, target = data_target

                if data_loader_creator.config.dataset_type == 'triplet':
                    data = data[:,0]
                    
                data = data.to(device)
                embedding, _ = model(data)
                embedding = embedding.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                if X is None:
                    X = embedding
                else:
                    X = np.append(X, embedding, axis=0)
                targets = np.append(targets, target)
    
    X_tsne = TSNE().fit_transform(X)

    dir_name = args.vis_base_dir + 'T' + str(task_id+1) + '/'
    makedirs(dir_name)

    for t_id, task in enumerate(data_loader_creator.tasks_targets):
        plt.figure()
        palette = sns.color_palette("bright", len(task))
        idx = np.isin(targets, task)
        sns_plot = sns.scatterplot(X_tsne[idx,0], X_tsne[idx,1], hue=targets[idx], legend='full', palette=palette, s=20)
        plt.savefig(dir_name + 'T' + str(t_id+1) + '.png')

    plt.figure()
    palette = sns.color_palette("bright", np.unique(targets).shape[0])
    sns_plot = sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=targets, legend='full', palette=palette, s=20)
    plt.savefig(dir_name + 'all.png')

    # tasks_targets = np.array(data_loader_creator.tasks_targets)
    tasks_targets = np.array([[0,1,2,3,4], [5,6,7,8,9]])
    tmp = np.zeros((tasks_targets.shape[0], targets.shape[0]))
    for i in range(tasks_targets.shape[0]):
        tmp[i] = np.isin(targets, tasks_targets[i])
    data_task = np.where(tmp.T == 1.)[1]
    plt.figure()
    palette = sns.color_palette("bright", len(tasks_targets))
    sns_plot = sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=data_task, legend='full', palette=palette, s=20)
    plt.savefig(dir_name + 'all_tasks.png')
    print('Visualization Ended.')

