import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import MultiHeadLinear


class ProjectiveWrapper(nn.Module):

    def __init__(self, model, output_dim):
        super(ProjectiveWrapper, self).__init__()
        self.model = model

        num_ftrs = model.classifier.in_features

        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(num_ftrs, output_dim)

    def forward(self, x):
        x = self.model.get_embedding(x)

        x = F.normalize(x, dim=1)

        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        
        return x

    def get_embedding(self, x):
        return self.forward(x)


class LinearWrapper(nn.Module):

    def __init__(self, model, num_classes):
        super(LinearWrapper, self).__init__()
        self.model = model

        num_ftrs = model.classifier.in_features
        self.classifier = MultiHeadLinear(num_ftrs, num_classes, no_grad=False)
        

    def forward(self, x):
        x = self.get_embedding(x)

        x = self.classifier(x)
        
        return x

    def get_embedding(self, x):
        x = self.model.get_embedding(x)

        x = F.normalize(x, dim=1)

        return x