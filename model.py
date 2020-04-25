import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_p = 0.5
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc(x)
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


class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()

        resnet34 = models.resnet34()
        
        num_feats = resnet34.fc.in_features

        self.feature_extractor = nn.Sequential(*list(resnet34.children())[:-1],
                                               nn.Flatten(),
                                               nn.Dropout(0.5))
        self.fc = nn.Linear(num_feats, num_classes)

    def forward(self, x):
        x = self.get_embedding(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return x, output

    def get_embedding(self, x):
        return self.feature_extractor(x)


def get_model(args):
    if args.dataset == 'mnist':
        model = Net()
    elif args.dataset == 'cifar100':
        model = ResNet34(num_classes=100)
    elif args.dataset == 'imagenet':
        model = ResNet34(num_classes=1000)
    else:
        raise ValueError('dataset is not supported.')
    return model