import math
import h5py
import numpy as np
from PIL import Image

import torch
import torch.utils.data.distributed
import torchvision


class DataLoaderConstructor:

    def __init__(self, args, train, **kwargs):
        self.train = train
        original_data, original_targets = self.get_data_targets(args.dataset)
        transforms = self.get_transforms(args.dataset)

        self.continual_constructor = ContinualIndexConstructor(args, original_targets, train)
        
        batch_size = args.batch_size if train else args.test_batch_size
        self.data_loaders = self.create_dataloaders(original_data,
                                                    original_targets, batch_size, **kwargs)

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
            
        if torch.is_tensor(dataset.targets):
            data = data.numpy()
            targets = targets.numpy()
        
        return data, targets

    def get_transforms(self, dataset_name):
        means = {
            'mnist':(0.13066051707548254,),
            'cifar10':(0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
            'cifar100':(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            'imagenet':(0.4814872465839461, 0.45771731927849263, 0.4082078035692402)
        }
        stds = {
            'mnist':(0.30810780244715075,),
            'cifar10':(0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
            'cifar100':(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            'imagenet':(0.2606993601638989, 0.2536456316098414, 0.2685610203190189)
        }

        transforms = []
        if dataste_name in ['cifar10', 'cifar100', 'imagenet']:
            transforms.extend([torchvision.transforms.RandomCrop(32, padding=4),
                                torchvision.transforms.RandomHorizontalFlip(),
                                torchvision.transforms.RandomRotation(15)])
        transforms.extend([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(means[dataste_name],
                                                             stds[dataste_name])])
        return torchvision.transforms.Compose(transforms)

    def create_dataloaders(self, data, targets, batch_size, **kwargs):
        data_loaders = []
        for task_indexes in self.continual_constructor.indexes:
            if args.model_type == 'softmax':
                dataset = SimpleDataset(data, targets, task_indexes, transform=self.transforms)
            elif args.model_type == 'triplet':
                dataset = TripletDataset(data, targets, task_indexes, transform=self.transforms)
            if args.distributed and self.train:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=False, sampler=sampler, **kwargs)
            data_loaders.append(data_loader)

        return data_loaders


class ContinualIndexConstructor:

    def __init__(self, args, targets, train):
        self.task_targets_set = self.create_task_targets_set(np.unique(targets), args.tasks)

        exemplar_size = args.exemplar_size if train else 0
        data_indexes, exemplars_indexes = self.divide_indexes_into_tasks(targets, exemplar_size)

        if args.oversample_ratio > 0.0:
            os_sizes = self.get_os_exemplar_size(data_indexes, exemplars_indexes, ratio)
            exemplars_indexes = self.get_oversampled_exemplars(exemplars_indexes, os_sizes)
        
        self.indexes = self.combine_data_exemplars(data_indexes, exemplars_indexes)

    def create_task_targets_set(self, unique_targets, ntasks):
        ntargets_per_task = int(len(unique_targets) / ntasks)
        ntargets_first_task = ntargets_per_task + len(unique_targets) % ntasks
        task_targets_set = [unique_targets[:ntargets_first_task]]

        target_idx = ntargets_first_task
        for i in range(ntasks-1):
            task_targets_set.append(unique_targets[target_idx: target_idx+ntargets_per_task])
            target_idx += ntargets_per_task

        return task_targets_set

    def divide_indexes_into_tasks(self, targets, exemplar_size):
        data_indexes = []
        exemplars_indexes = []

        for i, task_targets in enumerate(self.task_targets_set):
            prev_targets_set = []
            for prev_targets in self.task_targets_set[:i]:
                prev_targets_set.extend(prev_targets)

            task_data_indexes = np.empty((0), dtype=np.intc)
            task_exemplars_indexes = np.empty((0), dtype=np.intc)

            for target in task_targets:
                task_data_indexes = np.append(task_data_indexes, np.where(targets == target)[0])

            for target in prev_targets_set:
                size = int(exemplar_size/len(prev_targets_set))
                prev_all_indexes = np.where(targets == target)[0]
                idx = np.random.randint(prev_all_indexes.shape[0], size=size)
                exemplars_indexes = prev_all_indexes[idx]
                task_exemplars_indexes = np.append(task_exemplars_indexes, exemplars_indexes)

            data_indexes.append(task_data_indexes)
            exemplars_indexes.append(task_exemplars_indexes)
        
        return data_indexes, exemplars_indexes

    def get_os_exemplar_size(self, data_indexes, exemplars_indexes, ratio):
        os_sizes = []
        for i in range(len(exemplars_indexes)):
            data_target_size = len(self.task_targets_set[i])
            exemplar_target_size = sum([len(x) for x in self.task_targets_set[:i]])
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

            os_exemplar_indexes = exemplar_indexes[os_exemplars_idx]
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
        ntriplets = indexes.shape[0]

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
        return len(self.targets)

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