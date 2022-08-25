import sys
sys.path.insert(0, '/home/lpetrini/git/diffeo-sota/')

import torch
import torchvision
from torchvision import transforms

from datasets.twopoints import *
from .my_transforms import *

def load_cifar(p=500, resize=None, train=False, device='cpu', classes=None, shuffle=None,
               return_dataloader=False, gaussian_corruption_std=0):
    if shuffle is None:
        shuffle = not train
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    if resize is not None:
        test_list.append(transforms.Resize((resize, resize), interpolation=3))
    if gaussian_corruption_std:
        test_list.append(GaussianNoiseCorruption(std=gaussian_corruption_std))

    transform_test = transforms.Compose(test_list)

    testset = torchvision.datasets.CIFAR10(
        root='/home/lpetrini/data/cifar10', train=train, download=True, transform=transform_test)
    if classes is not None:
        testset = torch.utils.data.Subset(testset, [i for i, t in enumerate(testset.targets) if t in classes])
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=shuffle, num_workers=1)

    if not return_dataloader:
        imgs, y = next(iter(testloader))
        return imgs.to(device), y.to(device)
    else:
        return testloader


def load_svhn(p=500, resize=None, train=False):
    test_list = [
        transforms.ToTensor(),
        transforms.Normalize((0.4523, 0.4524, 0.4689), (0.2190, 0.2261, 0.2279)),
    ]
    if resize is not None:
        test_list.append(transforms.Resize((resize, resize), interpolation=3))

    transform_test = transforms.Compose(test_list)

    testset = torchvision.datasets.SVHN(
        root='/home/lpetrini/data/cifar10', split='train' if train else 'test', download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=1)

    imgs, y = next(iter(testloader))
    return imgs, y


def load_mnist(p=500, fashion=False, device='cpu'):
    if not fashion:
        testset = torchvision.datasets.MNIST(
            root='/home/lpetrini/data/mnist', train=False, download=True, transform=transforms.ToTensor())
    else:
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        testset = torchvision.datasets.FashionMNIST(
            root='/home/lpetrini/data/fashionmnist', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=False, num_workers=1)

    imgs, y = next(iter(testloader))
    return imgs.to(device), y.to(device)


def load_twopoints(p=500, train=False, device='cpu', shuffle=None, xi=14, gap=2, imsize=28,
                   norm='L2', pbc=False, local_translations=0, seed=0, return_dataloader=False,
                   bkg_noise=0, labelling='distance'):
    torch.manual_seed(seed)
    if shuffle is None:
        shuffle = not train
    testset = TwoPointsDataset(xi=xi, d=imsize, train=train, gap=gap, norm=norm, pbc=pbc, background_noise=bkg_noise, labelling=labelling)
    #     if local_translations:
    #         lt = torch.randint(2 * local_translations + 1, testset.coordinates.shape) - 1
    t = torch.randn(testset.coordinates.shape)
    t = (t / t.pow(2).sum(dim=-1, keepdim=True).sqrt() * local_translations).round()
    testset.coordinates += t.int()
    testset.coordinates = testset.coordinates.clip(0, imsize - 1)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=p, shuffle=shuffle, num_workers=1)
    if not return_dataloader:
        imgs, y = next(iter(testloader))
        return imgs.to(device), y.to(device)
    else:
        return testloader