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
        h, _ = self.model(x)
        # h = F.normalize(h, dim=1)

        o = self.l1(h)
        o = self.relu(o)
        o = self.l2(o)
        
        return h, o
