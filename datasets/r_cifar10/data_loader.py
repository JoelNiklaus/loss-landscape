"""
TODO Apply test transform from saved model

Use Test transform like this
    if 'test_transform' in checkpoint:
        self.transform = checkpoint['test_transform']
# Transform it
        img = self.transform(img)

"""

# Utils
import os

# Torch related stuff
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

from utils import get_relative_path
from datasets.r_cifar10.dataset import HydraImageFolder


def get_data_loaders_for_plotting(datapath='data', batch_size=128,
                                  threads=2, raw_data=False, data_split=1):
    """
    See: https://github.com/BayesWatch/cinic-10
    """
    data_folder = get_relative_path(datapath)
    cifar_mean = [0.4917877683632055, 0.48248474628523236, 0.4467221021943941]
    cifar_std = [0.24720946036868316, 0.24358763616111748, 0.26150537294105936]
    test_loader = data.DataLoader(
        HydraImageFolder('datasets/r_cifar10/data/CIFAR10/train',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=cifar_mean,
                                                                            std=cifar_std)])),
        batch_size=batch_size, shuffle=True)
    train_loader = test_loader

    return train_loader, test_loader


def get_data_loaders_for_training(args):
    data_folder = get_relative_path('data')
    cifar_mean = [0.4917877683632055, 0.48248474628523236, 0.4467221021943941]
    cifar_std = [0.24720946036868316, 0.24358763616111748, 0.26150537294105936]
    test_loader = data.DataLoader(
        HydraImageFolder(path='datasets/r_cifar10/data/CIFAR10/train',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=cifar_mean,
                                                                            std=cifar_std)])),
        batch_size=args.batch_size, shuffle=True)
    train_loader = data.DataLoader(
        HydraImageFolder(path='datasets/r_cifar10/data/CIFAR10/train',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                       transforms.Normalize(mean=cifar_mean,
                                                                            std=cifar_std)])),
        batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader
