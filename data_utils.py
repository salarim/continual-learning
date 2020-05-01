import torch
import torch.utils.data.distributed
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import math
import h5py


class SimpleDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], int(self.targets[idx])
        mode = 'L' if len(img.shape) == 2 else 'RGB'
        img = Image.fromarray(img, mode=mode)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class TripletDataset(torch.utils.data.Dataset):

    def __init__(self, data, targets, transform=None):
        self.transform = transform
        self.create_triplets(data, targets)

    def create_triplets(self, data, targets):
        ntriplets = targets.shape[0]

        anchor_idxs = np.empty(0, dtype=np.int)
        pos_idxs = np.empty(0, dtype=np.int)
        neg_idxs = np.empty(0, dtype=np.int)
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

        anchor = data[anchor_idxs]
        pos = data[pos_idxs]
        neg = data[neg_idxs]
        self.data = np.stack((anchor, pos, neg))
        self.targets = targets[anchor_idxs]

        
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        target = int(self.targets[idx])
        imgs = [self.data[0][idx], self.data[1][idx], self.data[2][idx]]
        for i in range(len(imgs)):
            mode = 'L' if len(imgs[i].shape) == 2 else 'RGB'
            imgs[i] = Image.fromarray(imgs[i], mode=mode)

            if self.transform is not None:
                imgs[i] = self.transform(imgs[i])

        return torch.stack((imgs[0], imgs[1], imgs[2])), target


class DataloaderCreator:
    def __init__(self, args, train, batch_size, shuffle, **kwargs):
        self.train = train
        data_dict = self.get_data_dict(args)
        exemplar_size = args.exemplar_size if train else 0
        self.create_task_target_set(list(data_dict.keys()), args.tasks)
        print('Data dictionary created!')
        data_list, target_list, exemplar_data_list, exemplar_target_list = self.get_longlife_data(data_dict, 
                            self.task_target_set,
                            exemplar_size)
        
        print('Task data and exemplars created!')
        self.data_loaders = []
        for i in range(len(target_list)):
            if args.model_type == 'softmax':
                dataset = SimpleDataset(data_list[i], target_list[i], transform=self.transform)
            elif args.model_type == 'triplet':
                dataset = TripletDataset(data_list[i], target_list[i], transform=self.transform)
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
            else:
                sampler = None
            data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs)
            self.data_loaders.append(data_loader)

        self.exemplar_datasets = []
        for i in range(len(exemplar_target_list)):
            if args.model_type == 'softmax':
                dataset = SimpleDataset(exemplar_data_list[i], exemplar_target_list[i],
                transform=self.transform)
            elif args.model_type == 'triplet':
                dataset = TripletDataset(exemplar_data_list[i], exemplar_target_list[i],
                transform=self.transform)
            self.exemplar_datasets.append(dataset)

        if self.train:
            bucket_size_list = []
            for data_loader in self.data_loaders:
                bucket_size_list.append(math.ceil(len(data_loader.dataset)/batch_size))
            if args.oversample:
                exemplar_size_list = [0] + [len(data_loader.dataset) for data_loader in self.data_loaders[1:]]
            else:
                exemplar_size_list = [0] + [exemplar_size]*(len(self.task_target_set)-1)
            self.buckets_list = self.distribute_exemplars(bucket_size_list, exemplar_size_list)
            print('Exemplars buckets list created!')

    
    def distribute_exemplars(self, bucket_size_list, exemplar_size_list):
        assert(len(bucket_size_list) == len(exemplar_size_list))
        assert(len(bucket_size_list) == len(self.exemplar_datasets))
        buckets_list = []

        for i in range(len(bucket_size_list)):
            dataset = self.exemplar_datasets[i]

            new_exemplars_idx = np.random.permutation(min(len(dataset), exemplar_size_list[i]))
            if exemplar_size_list[i] > len(dataset):
                extra_exemplar_idx = np.random.randint(len(dataset), size=exemplar_size_list[i]-len(dataset))
                new_exemplars_idx = np.append(new_exemplars_idx, extra_exemplar_idx)
            bucket_numbers = np.random.randint(bucket_size_list[i], size=exemplar_size_list[i])

            if exemplar_size_list[i] > 0:
                tmp_data, tmp_target = dataset[0]
                data_dtype, target_dtype = tmp_data.dtype, type(tmp_target)
                exemplars_data = torch.zeros((exemplar_size_list[i], 1) + tmp_data.shape[1:], dtype=data_dtype)
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

    def get_data_dict(self, args):
        means = {'mnist':(0.13066051707548254,),
         'cifar10':(0.49139967861519607, 0.48215840839460783, 0.44653091444546567),
         'cifar100':(0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
         'imagenet':(0.4814872465839461, 0.45771731927849263, 0.4082078035692402)}
        stds = {'mnist':(0.30810780244715075,),
         'cifar10':(0.2470322324632819, 0.24348512800005573, 0.26158784172796434),
         'cifar100':(0.2673342858792401, 0.2564384629170883, 0.27615047132568404),
         'imagenet':(0.2606993601638989, 0.2536456316098414, 0.2685610203190189)}

        if args.dataset == 'mnist':
            dataset = datasets.MNIST('./data/mnist', train=self.train, download=True)
        elif args.dataset == 'cifar10':
            dataset = datasets.CIFAR10('./data/cifar10', train=self.train, download=True)
        elif args.dataset == 'cifar100':
            dataset = datasets.CIFAR100('./data/cifar100', train=self.train, download=True)
        elif args.dataset == 'imagenet':
            if self.train:
                file_path = './data/imagenet/imagenet_train_500.h5'
            else:
                file_path = './data/imagenet/imagenet_test_100.h5'
            with h5py.File(file_path, 'r') as f:
                dataset = SimpleDataset(f['data'][:], f['labels'][:])
        else:
            raise ValueError('dataset is not supported.')
        
        if self.train:
            self.transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(means[args.dataset], stds[args.dataset])
                            ])
        else:
            self.transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(means[args.dataset], stds[args.dataset])
                            ])
        
        data = dataset.data
        targets = dataset.targets
        if torch.is_tensor(dataset.targets):
            data = data.numpy()
            targets = targets.numpy()
        unique_targets = np.unique(targets)
        data_dict = {}
        for target in unique_targets:
            idxs = targets == target
            data_dict[target] = data[idxs]
        
        return data_dict

    
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
        empty_data_shape = (0,) + tmp_data.shape[1:]

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

    
    def create_task_target_set(self, targets, ntasks):
        ntargets_per_task = int(len(targets) / ntasks)
        ntargets_first_task = ntargets_per_task + len(targets) % ntasks
        self.task_target_set = [targets[:ntargets_first_task]]
        target_idx = ntargets_first_task
        for i in range(ntasks-1):
            self.task_target_set.append(targets[target_idx: target_idx+ntargets_per_task])
            target_idx += ntargets_per_task
