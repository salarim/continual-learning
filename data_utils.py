import os
import cv2
import math
import h5py
import numpy as np
from PIL import Image

import torch
import torchvision


class DataConfig:

    def __init__(self, args, train, dataset, dataset_type, is_continual, batch_size, data_path=None,
                 workers=None,  tasks=None, exemplar_size=None, oversample_ratio=None):
        
        self.train = train
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.is_continual = is_continual
        self.batch_size = batch_size

        self.data_path = data_path if data_path else args.data_path
        self.workers = workers if workers else args.workers
        self.tasks = tasks if tasks else args.tasks
        self.exemplar_size = exemplar_size if exemplar_size else args.exemplar_size
        self.oversample_ratio = oversample_ratio if oversample_ratio else args.oversample_ratio


class DataLoaderConstructor:

    def __init__(self, config):
        self.config = config

        original_dataset, original_targets = self.get_dataset_targets(self.config.dataset)
        transforms = self.get_transforms(self.config.dataset)

        self.tasks_targets, indexes = \
            self.get_tasks_targets_indexes(original_targets)
        
        self.data_loaders = self.create_dataloaders(original_dataset, original_targets,
                                                    indexes, transforms)

    def get_dataset_targets(self, dataset_name):
        if dataset_name == 'mnist':
            dataset = torchvision.datasets.MNIST('./data/mnist',
                                                  train=self.config.train, download=True)
        elif dataset_name == 'cifar10':
            dataset = torchvision.datasets.CIFAR10('./data/cifar10',
                                                    train=self.config.train, download=True)
        elif dataset_name == 'cifar100':
            dataset = torchvision.datasets.CIFAR100('./data/cifar100',
                                                     train=self.config.train, download=True)
        elif dataset_name == 'imagenet':
            data_dir = os.path.join(self.config.data_path, 'train' if self.config.train else 'val')
            dataset = torchvision.datasets.ImageFolder(data_dir)
        else:
            raise ValueError('dataset is not supported.')
        
        targets = dataset.targets
        if torch.is_tensor(targets):
            targets = targets.numpy()
        elif type(targets) == list:
            targets = np.array(targets)

        return dataset, targets

    def get_transforms(self, dataset_name):
        means = {
            'mnist':(0.1307,),
            'cifar10':(0.485, 0.456, 0.406),
            'cifar100':(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
            'imagenet':(0.485, 0.456, 0.406)
        }
        stds = {
            'mnist':(0.3081,),
            'cifar10':(0.229, 0.224, 0.225),
            'cifar100':(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
            'imagenet':(0.229, 0.224, 0.225)
        }

        input_shape = {
            'mnist':(28, 28, 1),
            'cifar10':(32, 32, 3),
            'cifar100':(32, 32, 3),
            'imagenet':(224, 224, 4)
        }

        transforms = []
        if self.config.train:
            if dataset_name in ['cifar10', 'cifar100']:
                transforms.extend([torchvision.transforms.RandomResizedCrop(32),
                                    torchvision.transforms.RandomHorizontalFlip()])
            elif dataset_name == 'imagenet':
                transforms.extend([torchvision.transforms.RandomResizedCrop(224),
                                   torchvision.transforms.RandomHorizontalFlip()])
        else:
            if dataset_name == 'imagenet':
                transforms.extend([torchvision.transforms.Resize(256),
                                   torchvision.transforms.CenterCrop(224)])

        if self.config.dataset_type == 'contrastive':
            transforms.extend([torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)])
        
        transforms.extend([torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize(means[dataset_name],
                                                             stds[dataset_name])])
        
        return torchvision.transforms.Compose(transforms)

    def get_tasks_targets_indexes(self, original_targets):
        if self.config.is_continual:
            continual_constructor = ContinualIndexConstructor(original_targets, 
                                                              self.config.train, 
                                                              self.config.tasks, 
                                                              self.config.exemplar_size, 
                                                              self.config.oversample_ratio)
            tasks_targets = continual_constructor.tasks_targets
            indexes = continual_constructor.indexes
        else:
            tasks_targets = [list(np.unique(original_targets))] * self.config.tasks
            indexes = []
            for i in range(self.config.tasks):
                indexes.append(np.random.permutation(original_targets.shape[0]))
        
        return tasks_targets, indexes

    def create_dataloaders(self, org_dataset, targets, indexes, transforms):
        data_loaders = []

        for task_indexes in indexes:
            if self.config.dataset_type == 'softmax':
                dataset = SimpleDataset(org_dataset, task_indexes, transform=transforms)
            elif self.config.dataset_type == 'triplet':
                dataset = TripletDataset(org_dataset, targets, task_indexes, transform=transforms)
            elif self.config.dataset_type == 'contrastive':
                dataset = ContrastiveDataset(org_dataset, task_indexes, transform=transforms)
            
            kwargs = {'num_workers': self.config.workers, 'pin_memory': True} if \
                torch.cuda.device_count() > 0 else {}
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.config.batch_size, shuffle=True, **kwargs)
            data_loaders.append(data_loader)

        return data_loaders


class ContinualIndexConstructor:

    def __init__(self, targets, train, tasks, exemplar_size, oversample_ratio):
        self.tasks_targets = self.create_tasks_targets(np.unique(targets), tasks)

        data_indexes, exemplars_indexes = self.divide_indexes_into_tasks(targets, exemplar_size)

        if oversample_ratio > 0.0:
            os_sizes = self.get_os_exemplar_size(data_indexes, exemplars_indexes,
                                                 oversample_ratio)
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



class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, indexes, transform=None):
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform
        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        org_idx = self.indexes[idx]
        img, target = self.dataset.__getitem__(org_idx)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, targets, indexes, transform=None):
        self.dataset = dataset
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
        imgs = [self.dataset.__getitem__(anchor_indx)[0], 
                self.dataset.__getitem__(pos_idx)[0], 
                self.dataset.__getitem__(neg_idx)[0]]
        
        for i in range(len(imgs)):

            if self.transform is not None:
                imgs[i] = self.transform(imgs[i])

        return torch.stack((imgs[0], imgs[1], imgs[2])), target



class ContrastiveDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, indexes, transform):
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform
        
    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        org_idx = self.indexes[idx]
        img, target = self.dataset.__getitem__(org_idx)

        return self.transform(img), self.transform(img), target
