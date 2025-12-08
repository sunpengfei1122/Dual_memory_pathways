import os
from torchvision import datasets, transforms
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
def load_dataset1(dataset='MNIST', batch_size=100, dataset_path='../../data', is_cuda=False, num_workers=4):
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if is_cuda else {}
    if dataset == 'MNIST':
        num_classes = 10
        dataset_train = datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=True, download=True,
                                       transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=False, transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'FashionMNIST':
        num_classes = 10
        dataset_train = datasets.FashionMNIST(os.path.join(dataset_path, 'FashionMNIST'), train=True, download=False,
                                              transform=transforms.ToTensor())
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(os.path.join(dataset_path, 'FashionMNIST'), train=False,
                                  transform=transforms.ToTensor()),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'CIFAR10':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
        ])
        dataset_train = datasets.CIFAR10(os.path.join(dataset_path, 'CIFAR10'), train=True, download=False,
                                         transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(dataset_path, 'CIFAR10'), train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.424, 0.415, 0.384), (0.283, 0.278, 0.284))
                             ])),
            batch_size=batch_size, shuffle=False, **kwargs)

    elif dataset == 'SVHN':
        num_classes = 10
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
        ])
        dataset_train = torch.utils.data.ConcatDataset((
            datasets.SVHN(os.path.join(dataset_path, 'SVHN'), split='train', download=False, transform=train_transform),
            # datasets.SVHN('../data/SVHN', split='extra', download=True, transform=train_transform))
        ))
        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(os.path.join(dataset_path, 'SVHN'), split='test', download=False,
                          transform=transforms.Compose([
                              transforms.ToTensor(),
                              transforms.Normalize((0.431, 0.430, 0.446), (0.197, 0.198, 0.199))
                          ])),
            batch_size=batch_size, shuffle=False, **kwargs)
    else:
        raise Exception('No valid dataset is specified.')
    return train_loader, test_loader, num_classes

def npy_loader(path):
    sample = torch.from_numpy(np.load(path))
    return sample

class shdDataset(Dataset):
    # Characterizes a dataset for PyTorch
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.samples = np.loadtxt(self.data_paths+'.txt').astype('int')
        self.transform = transform

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):  
        inputindex = self.samples[index, 0]
        label = self.samples[index, 1]
        path = self.data_paths + '/'+str(inputindex.item()) + '.npy'
        input = torch.from_numpy(np.load(path))
        input = input.view(-1,140,5).sum(dim=2)
        label = torch.from_numpy(np.asarray(label))
        #if self.transform:
            
        #    x = self.transform(x)
        return input.float(), label

def load_dataset(dataset='MNIST', batch_size=100, dataset_path='../../data', is_cuda=False, num_workers=4):
    #kwargs = {'num_workers': num_workers, 'pin_memory': True} if is_cuda else {}
    #if True:
    num_classes = 20
    trainingSet = shdDataset('data/train')    
    dl_train = DataLoader(dataset=trainingSet, batch_size=128, shuffle=True, num_workers=0) 
    testSet = shdDataset('data/test')    
    dl_val = DataLoader(dataset=testSet, batch_size=128, shuffle=False, num_workers=0) 
    #    dataset_train = datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=True, download=True,
    #                                   transform=transforms.ToTensor())
    #    train_loader = torch.utils.data.DataLoader(
    #        dataset_train,
    #       # batch_size=batch_size, shuffle=True, **kwargs)
    #    test_loader = torch.utils.data.DataLoader(
    #        datasets.MNIST(os.path.join(dataset_path, 'MNIST'), train=False, transform=transforms.ToTensor()),
    #        batch_size=batch_size, shuffle=False, **kwargs
    return dl_train, dl_val, num_classes
