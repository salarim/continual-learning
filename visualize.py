import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from log_utils import makedirs
from models.nearest_prototype import NearestPrototype
from models.nearest_prototypes import NearestPrototypes

try:
    from tsnecuda import TSNE
except ImportError:
    from sklearn.manifold import TSNE


def plot_embedding_tsne(args, task_id, data_loader_creator, model, device, nearest_proto_model=None):
    X_tsne, targets, outputs, protos_tsne, protos_targets = extract_tsne_features(data_loader_creator, 
                                                                model, device, nearest_proto_model)

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
    if nearest_proto_model is not None:
        sns_plot = sns.scatterplot(protos_tsne[:,0], protos_tsne[:,1], hue=protos_targets, legend=False, palette=palette, s=30, marker='X', edgecolor='black')
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

    
    plt.figure()
    palette = sns.color_palette("bright", np.unique(outputs).shape[0])
    sns_plot = sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=outputs, legend='full', palette=palette, s=20)
    if nearest_proto_model is not None:
        sns_plot = sns.scatterplot(protos_tsne[:,0], protos_tsne[:,1], hue=protos_targets, legend=False, palette=palette, s=30, marker='X', edgecolor='black')
    plt.savefig(dir_name + 'preds.png')

    plt.figure()
    palette = sns.color_palette("Greys", 2)
    sns_plot = sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=(outputs==targets).astype(int)-2, legend='full', palette=palette, s=20)
    if nearest_proto_model is not None:
        palette = sns.color_palette("bright", np.unique(protos_targets).shape[0])
        sns_plot = sns.scatterplot(protos_tsne[:,0], protos_tsne[:,1], hue=protos_targets, legend='full', palette=palette, s=30, marker='X')
    plt.savefig(dir_name + 'correctness.png')

    print('Visualization Ended.')

def extract_tsne_features(data_loader_creator, model, device, nearest_proto_model=None):
    X = None
    outputs = None
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
                embedding, output = model(data)
                output = torch.argmax(output, dim=1)
                if nearest_proto_model is not None:
                    output = nearest_proto_model.predict(embedding)
                embedding = embedding.cpu().detach().numpy()
                output = output.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                if X is None:
                    X = embedding
                    outputs = output
                else:
                    X = np.append(X, embedding, axis=0)
                    outputs = np.append(outputs, output, axis=0)
                targets = np.append(targets, target)

    protos = np.empty((0, X.shape[1]))
    protos_targets = np.empty((0))
    if nearest_proto_model is not None:
        last_task = max(nearest_proto_model.task_class_prototypes.keys(), default=-1)
        for target, feats in nearest_proto_model.task_class_prototypes[last_task].items():
            if isinstance(nearest_proto_model, NearestPrototype):
                feats = feats[0].cpu().detach().numpy()
                protos = np.append(protos, feats)
                protos_targets = np.append(protos_targets, target)
            elif isinstance(nearest_proto_model, NearestPrototypes):
                feats = feats.cpu().detach().numpy()
                protos = np.append(protos, feats, axis=0)
                target_targets = np.full((feats.shape[0]), target)
                protos_targets = np.append(protos_targets, target_targets)

    X = np.append(X, protos, axis=0)

    X_tsne = TSNE().fit_transform(X)

    protos_tsne = X_tsne[-protos.shape[0]:]
    X_tsne = X_tsne[:-protos.shape[0]]

    return X_tsne, targets, outputs, protos_tsne, protos_targets
