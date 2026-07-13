import torch
import torch.utils.data as Data
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import parameters
from utils import Cutout    

def _build_train_transform(use_cutout):
    if use_cutout:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4913, 0.4821, 0.4465],std=[0.2470, 0.2435, 0.2616]),
            Cutout(n_holes=1, length=16)  # 添加 Cutout 数据增强
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),  #先四周填充0，在把图像随机裁剪成32*32
            transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4913, 0.4821, 0.4465],std=[0.2470, 0.2435, 0.2616])
        ])
    return transform_train


def Train_data_Loader(params=None):
    params = params or parameters.get_parameters()
    transform_train = _build_train_transform(params.use_cutout)

    transform_valid = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4913, 0.4821, 0.4465],std=[0.2470, 0.2435, 0.2616])
    ])

    data_dir = params.data_dir

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform_train)

    validset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform_valid)

    valid_size = params.valid_size
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = Data.DataLoader(
            dataset=trainset,  
            batch_size=params.batch_size, 
            sampler=train_sampler,  
            num_workers=params.num_workers,
        )

    valid_loader = Data.DataLoader(
            dataset=validset,  
            batch_size=params.batch_size, 
            sampler=valid_sampler,  
            num_workers=params.num_workers,
        )

    return train_loader, valid_loader


def Full_train_data_Loader(params=None):
    params = params or parameters.get_parameters()
    transform_train = _build_train_transform(params.use_cutout)

    trainset = torchvision.datasets.CIFAR10(
        root=params.data_dir,
        train=True,
        download=True,
        transform=transform_train
    )

    train_loader = Data.DataLoader(
        dataset=trainset,
        batch_size=params.batch_size,
        shuffle=True,
        num_workers=params.num_workers,
    )

    return train_loader


def Test_data_Loader(params=None):
    params = params or parameters.get_parameters()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4913, 0.4821, 0.4465],std=[0.2470, 0.2435, 0.2616])
    ])

    data_dir = params.data_dir

    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                        download=True, transform=transform)

    test_loader = Data.DataLoader(
            dataset=testset,  
            batch_size=params.batch_size, 
            shuffle=False,  
            num_workers=params.num_workers,
        )
 
    return test_loader