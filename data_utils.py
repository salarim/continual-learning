import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import math

class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        img = Image.fromarray(img, mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class DataloaderCreator:
    def __init__(self, train, batch_size, shuffle, **kwargs):
        self.train = train
        mnist_dict = self.get_mnist_dict()
        exemplar_size = 100 if train else 0
        self.task_target_set = [[0,1,2,3], [4,5], [6,7], [8,9]]
        data_list, target_list, exemplar_data_list, exemplar_target_list = self.get_longlife_data(mnist_dict, 
                            self.task_target_set,
                            exemplar_size)

        self.data_loaders = []
        transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
        for i in range(len(target_list)):
            dataset = SimpleDataset(data_list[i], target_list[i], transform=transform)
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
            self.data_loaders.append(data_loader)

        self.exemplar_datasets = []
        for i in range(len(exemplar_target_list)):
            dataset = SimpleDataset(exemplar_data_list[i], exemplar_target_list[i],
             transform=transform)
            self.exemplar_datasets.append(dataset)

        if self.train:
            bucket_size_list = []
            for data_loader in self.data_loaders[1:len(self.data_loaders)]:
                bucket_size_list.append(math.ceil(len(data_loader.dataset)/batch_size))
            # exemplar_size_list = [len(data_loader.dataset) for data_loader in self.data_loaders[1:]]
            exemplar_size_list = [exemplar_size]*3
            self.buckets_list = exemplar_buckets_list = self.distribute_exemplars(bucket_size_list,
             exemplar_size_list)

    
    def distribute_exemplars(self, bucket_size_list, exemplar_size_list):
        assert(len(bucket_size_list) == len(exemplar_size_list))
        assert(len(bucket_size_list) == len(self.exemplar_datasets))
        buckets_list = []

        for i in range(len(bucket_size_list)):
            dataset = self.exemplar_datasets[i]
            tmp_data, tmp_target = dataset[0]
            data_dtype, target_dtype = tmp_data.dtype, type(tmp_target)

            new_exemplars_idx = np.random.permutation(min(len(dataset), exemplar_size_list[i]))
            if exemplar_size_list[i] > len(dataset):
                extra_exemplar_idx = np.random.randint(len(dataset), size=exemplar_size_list[i]-len(dataset))
                new_exemplars_idx = np.append(new_exemplars_idx, extra_exemplar_idx)
            bucket_numbers = np.random.randint(bucket_size_list[i], size=exemplar_size_list[i])
            exemplars_data = torch.zeros(exemplar_size_list[i], 1, tmp_data.shape[1], tmp_data.shape[2],
             dtype=data_dtype)
            exemplars_target = torch.zeros(exemplar_size_list[i], dtype=target_dtype)
            for j, idx in enumerate(new_exemplars_idx):
                exemplars_data[j] = dataset[idx][0]
                exemplars_target[j] = dataset[idx][1]

            buckets = {}
            for bucket_number in range(bucket_size_list[i]):
                buckets[bucket_number] = (None,None)
                bucket_idx = bucket_numbers == bucket_number
                if np.any(bucket_idx):
                    buckets[bucket_number] = exemplars_data[bucket_idx], exemplars_target[bucket_idx]
            buckets_list.append(buckets)
        
        return buckets_list

    def get_mnist_dict(self):
        dataset = datasets.MNIST('../data', train=self.train, download=True)
        self.transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])
        targets = dataset.targets.numpy()
        data = dataset.data.numpy()
        unique_targets = np.unique(targets)
        mnist_dict = {}
        for target in unique_targets:
            idxs = targets == target
            mnist_dict[target] = data[idxs]
        
        return mnist_dict

    
    def get_longlife_data(self, data_dict, task_target_set, exemplar_size):
        """
        Args:
            data_dict (dictionary): A dictionary with targets as keys and data as values.
            tasks_target_set (list): A list of targets in each tasks.
                e.g. [[0,1,2,3], [4,5], [6,7], [8,9]]
            exemplar_size (int): The size of exemplars that should be chosen from previous tasks.
        """

        data_list = []
        target_list = []
        exemplar_data_list = []
        exemplar_target_list = []

        tmp_target, tmp_data = next(iter(data_dict.items()))
        data_dtype, target_dtype = tmp_data.dtype, tmp_target.dtype
        empty_data_shape = (0, tmp_data.shape[1], tmp_data.shape[2])

        for i, task_targets in enumerate(task_target_set):
            targets = np.empty((0), dtype=target_dtype)
            data = np.empty(empty_data_shape, dtype=data_dtype)

            for target in task_targets:
                new_targets = np.full((len(data_dict[target])), target, dtype=target_dtype)
                new_data = data_dict[target]
                targets = np.append(targets, new_targets)
                data = np.append(data, new_data, axis=0)

            prev_targets_set = []
            for prev_targets in task_target_set[:i]:
                prev_targets_set.extend(prev_targets)

            if exemplar_size > 0 and len(prev_targets_set) > 0:
                prev_targets_all = np.empty((0), dtype=target_dtype)
                prev_data_all = np.empty(empty_data_shape, dtype=data_dtype)
                for prev_target in prev_targets_set:
                    size = int(exemplar_size/len(prev_targets_set))
                    prev_targets = np.full(size, prev_target, dtype=target_dtype)
                    prev_all_data = data_dict[prev_target]
                    idx = np.random.randint(prev_all_data.shape[0], size=size)
                    prev_data = prev_all_data[idx,:,:]
                    prev_targets_all = np.append(prev_targets_all, prev_targets)
                    prev_data_all = np.append(prev_data_all, prev_data, axis=0)
                exemplar_target_list.append(prev_targets_all)
                exemplar_data_list.append(prev_data_all)

            perm = np.random.permutation(len(targets))
            targets = targets[perm]
            data = data[perm]

            data_list.append(data)
            target_list.append(targets)
        
        return data_list, target_list, exemplar_data_list, exemplar_target_list
