import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.dropout_p = 0.5
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.classifier = nn.Linear(128, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.classifier(x)
        output = F.log_softmax(x, dim=1)
        return x, output

    def get_embedding(self, x):
        x = F.dropout(self.conv1(x), training=True, p=self.dropout_p)
        x = F.relu(x)
        x = F.dropout(self.conv2(x), training=True, p=self.dropout_p)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.dropout(self.fc1(x), training=True, p=self.dropout_p)
        x = F.relu(x)
        return x
