import torch
import numpy as np
from tsnecuda import TSNE
import seaborn as sns


def plot_embedding_tsne(data_loader_creator, model, device):
    embedding_size = model.fc2.in_features
    X = np.empty((0,embedding_size), dtype=np.float32)
    targets = np.empty((0))
    with torch.no_grad():
        for data_loader in data_loader_creator.data_loaders:
            for data, target in data_loader:
                data = data.to(device)
                embedding = model.get_embedding(data)
                embedding = embedding.cpu().detach().numpy()
                target = target.cpu().detach().numpy()
                X = np.append(X, embedding, axis=0)
                targets = np.append(targets, target)
    
    X_tsne = TSNE().fit_transform(X)
    print(X_tsne.shape)
    palette = sns.color_palette("bright", 10)
    sns_plot = sns.scatterplot(X_tsne[:,0], X_tsne[:,1], hue=targets, legend='full', palette=palette)
    sns_plot.figure.savefig("plots/output.png")

