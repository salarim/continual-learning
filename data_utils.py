import math
import h5py
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
import torchvision


class DataLoaderConstructor:

    def __init__(self, args, dataset, train, is_continual):
        self.train = train
        original_data, original_targets = self.get_data_targets(dataset)
        transforms = self.get_transforms(dataset)

        self.tasks_targets, indexes = \
            self.get_tasks_targets_indexes(args, original_targets, is_continual)
        
        self.data_loaders = self.create_dataloaders(args, original_data, original_targets
                                                    indexes, transforms)

    def get_data_targets(self, dataset_name):
        if dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST('./data/mnist',
                                                  train=self.train, download=True)
            data, targets = dataset.data, dataset.targets
        elif dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10('./data/cifar10',
                                                    train=self.train, download=True)
            data, targets = dataset.data, dataset.targets
        elif dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100('./data/cifar100',
                                                     train=self.train, download=True)
            data, targets = dataset.data, dataset.targets
        elif dataset_name == 'imagenet':
            if self.train:
                file_path = './data/imagenet/imagenet_train_500.h5'
            else:
                file_path = './data/imagenet/imagenet_test_100.h5'
            with h5py.File(file_path, 'r') as f:
                data, targets = f['data'][:], f['labels'][:]
        else:
            raise ValueError('dataset is not supported.')
            
        if torch.is_tensor(targets):
            data = data.numpy()
            targets = targets.numpy()
        
        return data, targets

    def get_transforms(self, dataset_name):
        means = {
            'mnist':(0.1307,),
            'cifar10':(0.485, 0.456, 0.406),
            'cifar100':(0.4914, 0.4822, 0.4465),
            'imagenet':(0.485, 0.456, 0.406)
        }
        stds = {
            'mnist':(0.3081,),
            'cifar10':(0.229, 0.224, 0.225),
            'cifar100':(0.2023, 0.1994, 0.2010),
            'imagenet':(0.229, 0.224, 0.225)
        }

        transforms = []
        if dataset_name in ['cifar10', 'cifar100', 'imagenet'] and self.train:
            transforms.extend([torchvision.transforms.RandomCrop(32, padding=4),
                                torchvision.transforms.RandomHorizontalFlip()])
        transforms.extend([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(means[dataset_name],
                                                             stds[dataset_name])])
        return torchvision.transforms.Compose(transforms)

    def get_tasks_targets_indexes(self, args, original_targets, is_continual):
        if is_continual:
            continual_constructor = ContinualIndexConstructor(args, original_targets, train)
            tasks_targets = continual_constructor.tasks_targets
            indexes = continual_constructor.indexes
        else:
            tasks_targets = [list(np.unique(original_targets))] * args.tasks
            indexes = []
            for i in range(args.tasks)
            indexes.append(np.random.permutation(original_targets.shape[0]))
        
        return tasks_targets, indexes

    def create_dataloaders(self, args, data, targets, indexes, transforms):
        data_loaders = []

        batch_size = args.batch_size if self.train else args.test_batch_size
        for task_indexes in indexes:
            if args.model_type == 'softmax':
                dataset = SimpleDataset(data, targets, task_indexes, transform=transforms)
            elif args.model_type == 'triplet':
                dataset = TripletDataset(data, targets, task_indexes, transform=transforms)
            elif args.model_type == 'contrastive':
                dataset = ContrastiveDataset(data, targets, task_indexes, transform=transforms)

            if args.distributed and self.train:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None
            
            kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.gpu else {}
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
            data_loaders.append(data_loader)

        return data_loaders


class ContinualIndexConstructor:

    def __init__(self, args, targets, train):
        self.tasks_targets = self.create_tasks_targets(np.unique(targets), args.tasks)

        exemplar_size = args.exemplar_size if train else 0
        data_indexes, exemplars_indexes = self.divide_indexes_into_tasks(targets, exemplar_size)

        if args.oversample_ratio > 0.0:
            os_sizes = self.get_os_exemplar_size(data_indexes, exemplars_indexes,
                                                 args.oversample_ratio)
            exemplars_indexes = self.get_oversampled_exemplars(exemplars_indexes, os_sizes)
        
        self.indexes = self.combine_data_exemplars(data_indexes, exemplars_indexes)

    def create_tasks_targets(self, unique_targets, ntasks):
        ntargets_per_task = int(len(unique_targets) / ntasks)
        ntargets_first_task = ntargets_per_task + len(unique_targets) % ntasks
        tasks_targets = [unique_targets[:ntargets_first_task]]

        target_idx = ntargets_first_task
        for i in range(ntasks-1):
            tasks_targets.append(unique_targets[target_idx: target_idx+ntargets_per_task])
            target_idx += ntargets_per_task

        return tasks_targets

    def divide_indexes_into_tasks(self, targets, exemplar_size):
        data_indexes = []
        exemplars_indexes = []

        for i, task_targets in enumerate(self.tasks_targets):
            prev_targets = []
            for prev_tasks_targets in self.tasks_targets[:i]:
                prev_targets.extend(prev_tasks_targets)

            task_data_indexes = np.empty((0), dtype=np.intc)
            task_exemplars_indexes = np.empty((0), dtype=np.intc)

            for target in task_targets:
                task_data_indexes = np.append(task_data_indexes, np.where(targets == target)[0])

            for target in prev_targets:
                size = int(exemplar_size/len(prev_targets))
                prev_all_indexes = np.where(targets == target)[0]
                idx = np.random.randint(prev_all_indexes.shape[0], size=size)
                target_exemplars_indexes = prev_all_indexes[idx]
                task_exemplars_indexes = np.append(task_exemplars_indexes, target_exemplars_indexes)

            data_indexes.append(task_data_indexes)
            exemplars_indexes.append(task_exemplars_indexes)
        
        return data_indexes, exemplars_indexes

    def get_os_exemplar_size(self, data_indexes, exemplars_indexes, ratio):
        os_sizes = []
        for i in range(len(exemplars_indexes)):
            data_target_size = len(self.tasks_targets[i])
            exemplar_target_size = sum([len(x) for x in self.tasks_targets[:i]])
            data_size = data_indexes[i].shape[0]

            size = int(exemplar_target_size * ratio * (data_size / data_target_size))
            if exemplars_indexes[i].shape[0] == 0:
                size = 0
            os_sizes.append(size)
        return os_sizes
    
    def get_oversampled_exemplars(self, exemplars_indexes, os_sizes):
        os_exemplars_indexes = []

        for i, exemplar_indexes in enumerate(exemplars_indexes):
            os_exemplars_indexes_idx = np.random.permutation(min(len(exemplar_indexes),
                                                             os_sizes[i]))
            if os_sizes[i] > len(exemplar_indexes):
                extra_exemplar_indexes_idx = np.random.randint(len(exemplar_indexes),
                                                       size=os_sizes[i]-len(exemplar_indexes))
                os_exemplars_indexes_idx = np.append(os_exemplars_indexes_idx,
                                                     extra_exemplar_indexes_idx)

            os_exemplar_indexes = exemplar_indexes[os_exemplars_indexes_idx]
            os_exemplars_indexes.append(os_exemplar_indexes)
        
        return os_exemplars_indexes

    def combine_data_exemplars(self, data_indexes, exemplars_indexes):
        indexes = []

        for i in range(len(data_indexes)):
            task_data_indexes = data_indexes[i]
            task_exemplars_indexes = exemplars_indexes[i]

            task_indexes = np.append(task_data_indexes, task_exemplars_indexes)
            perm = np.random.permutation(len(task_indexes))
            task_indexes = task_indexes[perm]

            indexes.append(task_indexes)
        
        return indexes



class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, indexes, transform=None):
        self.data = data
        self.targets = targets
        self.indexes = indexes
        self.transform = transform
        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        org_idx = self.indexes[idx]
        img, target = self.data[org_idx], int(self.targets[org_idx])
        mode = 'L' if len(img.shape) == 2 else 'RGB'
        img = Image.fromarray(img, mode=mode)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, indexes, transform=None):
        self.data = data
        self.targets = targets
        self.indexes = indexes
        self.transform = transform
        self.anchor_idxs, self.pos_idxs, self.neg_idxs = self.create_triplets(targets, indexes)

    def create_triplets(self, targets, indexes):
        targets = targets[indexes]

        anchor_idxs = np.empty(0, dtype=np.intc)
        pos_idxs = np.empty(0, dtype=np.intc)
        neg_idxs = np.empty(0, dtype=np.intc)
        for target in np.unique(targets):
            anchor_idx = np.where(targets==target)[0]
            pos_idx = np.where(targets==target)[0]
            while np.any((anchor_idx-pos_idx)==0):
                np.random.shuffle(pos_idx)
            neg_idx = np.random.choice(np.where(targets!=target)[0], len(anchor_idx), replace=True)
            anchor_idxs = np.append(anchor_idxs, anchor_idx)
            pos_idxs = np.append(pos_idxs, pos_idx)
            neg_idxs = np.append(neg_idxs, neg_idx)

        perm = np.random.permutation(len(anchor_idxs))
        anchor_idxs = anchor_idxs[perm]
        pos_idxs = pos_idxs[perm]
        neg_idxs = neg_idxs[perm]

        return anchor_idxs, pos_idxs, neg_idxs

        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        anchor_indx = self.indexes[self.anchor_idxs[idx]]
        pos_idx = self.indexes[self.pos_idxs[idx]]
        neg_idx = self.indexes[self.neg_idxs[idx]]

        target = int(self.targets[anchor_indx])
        imgs = [self.data[anchor_indx], self.data[pos_idx], self.data[neg_idx]]
        for i in range(len(imgs)):
            mode = 'L' if len(imgs[i].shape) == 2 else 'RGB'
            imgs[i] = Image.fromarray(imgs[i], mode=mode)

            if self.transform is not None:
                imgs[i] = self.transform(imgs[i])

        return torch.stack((imgs[0], imgs[1], imgs[2])), target



class ContrastiveDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, indexes, transform):
        self.data = data
        self.targets = targets
        self.indexes = indexes
        self.transform = transform
        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        org_idx = self.indexes[idx]
        img, target = self.data[org_idx], int(self.targets[org_idx])
        mode = 'L' if len(img.shape) == 2 else 'RGB'
        img = Image.fromarray(img, mode=mode)

        return self.transform(img), self.transform(img), target
