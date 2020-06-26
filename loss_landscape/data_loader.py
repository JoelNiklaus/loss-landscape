import importlib

import torch
import os
import argparse


def load_dataset(dataset='cifar10', datapath='datasets/cifar10/data', batch_size=128,
                 threads=2, raw_data=False, data_split=1, split_idx=0,
                 trainloader_path="", testloader_path=""):
    """
    Setup dataloader. The data is not randomly cropped as in training because
    we want to estimate the loss value with a fixed dataset.

    Args:
        dataset:
        datapath:
        raw_data: raw images, no data preprocessing
        data_split: the number of splits for the training dataloader
        split_idx: the index for the split of the dataloader, starting at 0
        trainloader_path:
        testloader_path:

    Returns:
        train_loader, test_loader
    """

    # use specific dataloaders
    if trainloader_path and testloader_path:
        assert os.path.exists(trainloader_path), 'trainloader does not exist'
        assert os.path.exists(testloader_path), 'testloader does not exist'
        train_loader = torch.load(trainloader_path)
        test_loader = torch.load(testloader_path)
        return train_loader, test_loader

    assert split_idx < data_split, 'the index of data partition should be smaller than the total number of split'

    module = 'datasets.' + dataset + '.data_loader'
    mymod = importlib.import_module(module)  # import the module: same as import datasets.{dataset}.dataloader

    load_function = getattr(mymod, "get_data_loaders_for_plotting")
    return load_function(datapath, batch_size, threads, raw_data, data_split)


###############################################################
####                        MAIN
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--mpi', '-m', action='store_true', help='use mpi')
    parser.add_argument('--cuda', '-c', action='store_true', help='use cuda')
    parser.add_argument('--threads', default=2, type=int, help='number of threads')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--dataset', default='cifar10', help='cifar10 | imagenet')
    parser.add_argument('--datapath', default='data', metavar='DIR', help='path to the dataset')
    parser.add_argument('--raw_data', action='store_true', default=False, help='do not normalize data')
    parser.add_argument('--data_split', default=1, type=int, help='the number of splits for the dataloader')
    parser.add_argument('--split_idx', default=0, type=int, help='the index of data splits for the dataloader')
    parser.add_argument('--trainloader', default='', help='path to the dataloader with random labels')
    parser.add_argument('--testloader', default='', help='path to the testloader with random labels')

    args = parser.parse_args()

    trainloader, testloader = load_dataset(args.dataset, args.datapath,
                                           args.batch_size, args.threads, args.raw_data,
                                           args.data_split, args.split_idx,
                                           args.trainloader, args.testloader)

    print('num of batches: %d' % len(trainloader))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        print('batch_idx: %d   batch_size: %d' % (batch_idx, len(inputs)))
