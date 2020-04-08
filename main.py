from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from PIL import Image
import numpy as np
from termcolor import cprint

class MyMnist(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train = train
        dataset = datasets.MNIST('../data', train=train, download=True)
        self.transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
        targets = dataset.targets.numpy()
        data = dataset.data.numpy()

        all_labels = np.unique(targets)
        data_dict = {}
        for label in all_labels:
            idxs = targets == label
            data_dict[label] = data[idxs]

        self.targets = np.empty((0), dtype=targets.dtype)
        self.data = np.empty((0, data.shape[1], data.shape[2]), dtype=data.dtype)
        
        self.get_longlife_data(data_dict, [], range(10), targets.dtype)
        # self.get_seprated_data(data_dict, [0,1,2,3,4], range(10), targets.dtype)
        # self.data = data
        # self.targets = targets


    def get_seprated_data(self, data_dict, train_labels, test_labels, dtype):
        if self.train:
            for label in train_labels:
                new_targets = np.full((len(data_dict[label])), label, dtype=dtype)
                new_data = data_dict[label]
                self.targets = np.append(self.targets, new_targets)
                self.data = np.append(self.data, new_data, axis=0)
            perm = np.random.permutation(len(self.targets))
            self.targets = self.targets[perm]
            self.data = self.data[perm]
        else:
            for label in test_labels:
                new_targets = np.full((len(data_dict[label])), label, dtype=dtype)
                new_data = data_dict[label]
                self.targets = np.append(self.targets, new_targets)
                self.data = np.append(self.data, new_data, axis=0)

    def get_unbalanced_data(self, data_dict, dtype):
        data_size = {0:data_dict[0].shape[0]}
        for label in range(1,10):
            data_size[label] = int(data_size[label-1] / 2)
        print(data_size)
        
        for label in range(10):
            new_targets = np.full((data_size[label]), label, dtype=dtype)
            new_data = data_dict[label][:data_size[label]]
            
            self.targets = np.append(self.targets, new_targets)
            self.data = np.append(self.data, new_data, axis=0)

    def get_longlife_data(self, data_dict, base_labels, tasks_labels, dtype, add_pre_samples=True):
        # Add base labels
        for label in base_labels:
            new_targets = np.full((len(data_dict[label])), label, dtype=dtype)
            new_data = data_dict[label]
            self.targets = np.append(self.targets, new_targets)
            self.data = np.append(self.data, new_data, axis=0)
        perm = np.random.permutation(len(self.targets))
        self.targets = self.targets[perm]
        self.data = self.data[perm]

        # Add tasks labels
        for i in range(len(tasks_labels)):
            label = tasks_labels[i]
            new_targets = np.full((len(data_dict[label])), label, dtype=dtype)
            new_data = data_dict[label]

            # add samples from previous classes
            if self.train and add_pre_samples:
                total_prev_size = 100
                prev_labels = list(base_labels) + list(tasks_labels[:i])
                for prev_label in prev_labels:
                    prev_size = min(int(total_prev_size / len(prev_labels)), len(data_dict[prev_label]))

                    prev_targets = np.full((prev_size), prev_label, dtype=dtype)
                    prev_data = data_dict[prev_label][:prev_size]

                    new_targets = np.append(new_targets, prev_targets)
                    new_data = np.append(new_data, prev_data, axis=0)
                
            perm = np.random.permutation(len(new_targets))
            new_targets = new_targets[perm]
            new_data = new_data[perm]

            self.targets = np.append(self.targets, new_targets)
            self.data = np.append(self.data, new_data, axis=0)

        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.dropout_p = 0.5
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.dropout(self.conv1(x), training=True, p=self.dropout_p)
        x = F.relu(x)
        x = F.dropout(self.conv2(x), training=True, p=self.dropout_p)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.dropout(self.fc1(x), training=True, p=self.dropout_p)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, test_loader, optimizer, epoch):
    T = 10
    model.train()
    max_target = -1
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output_list = []
        for i in range(T):
            output_list.append(torch.unsqueeze(model(data), 0))
        output_mean = torch.cat(output_list, 0).mean(0)
        output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
        output_entropy = (-output_mean.exp() * output_mean).sum(dim=1).mean().item()
        loss = F.nll_loss(output_mean, target)
        loss.backward()
        # Change lr
        scaled_entropy = output_entropy * 100.
        new_lr = args.lr / min(max(scaled_entropy, 1.0), 100.0)
        print('New Learning Rate: {:.5f}'.format(new_lr))
        for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        # max_target = max(max_target, max(target).item())
        # if min(target).item() < max_target:
        #     new_lr = args.lr
        # else:
        #     new_lr = args.lr / 10
        # print('New Learning Rate: {:.5f}'.format(new_lr))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lr
        #
        optimizer.step()

        pred = output_mean.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()

        if batch_idx % args.log_interval == 0:
            print('Batch labels: ' + str(torch.unique(target).tolist()))
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} Batch_Acc: {:.2f} Entropy: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), correct / target.shape[0],
                output_entropy))

            # test(args, model, device, test_loader, print_entropy=False)


def test(args, model, device, test_loader, print_entropy=True):
    model.eval()
    T = 1
    test_loss = 0
    correct = 0
    label_correct = {}
    label_all = {}
    output_variances = {i:[] for i in range(10)}
    output_entropies = {i:[] for i in range(10)}

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output_list = []
            for i in range(T):
                output_list.append(torch.unsqueeze(model(data), 0))
            output_mean = torch.cat(output_list, 0).mean(0)
            output_variance = torch.cat(output_list, 0).var(dim=0).mean().item()
            output_entropy = (-output_mean.exp() * output_mean).sum(dim=1).mean().item()
            test_loss += F.nll_loss(output_mean, target, reduction='sum').item()  # sum up batch loss
            pred = output_mean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            for label in range(10):
                inds = (target == label)
                corr = pred[inds,:].eq(target[inds].view_as(pred[inds,:])).sum().item()
                if label not in label_correct:
                    label_correct[label] = 0
                    label_all[label] = 0
                label_correct[label] += corr
                label_all[label] += inds.sum().item()

            for label in target.unique().tolist():
                output_variances[label].append(output_variance)
                output_entropies[label].append(output_entropy)
            # print('Test labels:', target.unique().tolist(),
            # 'Var:', output_variance, 'Entropy:', output_entropy)

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    for label in range(10):
        print('{:4d}: {:4.0f}%'.format(label, 100. * label_correct[label]/label_all[label]), end=' ')
    print('\n')
    if print_entropy:
        for label in range(10):
            print(label, np.mean(output_variances[label]), np.mean(output_entropies[label]))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        MyMnist(train=True),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        MyMnist(train=False),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, test_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        # scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()
