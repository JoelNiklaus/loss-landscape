import torch
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np

def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)

def get_data_loaders_for_plotting(datapath='data', batch_size=128,
                                  threads=2, raw_data=False, data_split=1):
    """
    Gets the data loaders. This method is called dynamically from data_loader.py in the root directory.
    Args:
       datapath:
       batch_size:
       threads:
       raw_data:
       data_split:

    Returns:         A tuple containing the trainloader and the testloader.

    """

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    data_folder = get_relative_path(datapath)
    if raw_data:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                            download=True, transform=transform)
    # If data_split>1, then randomly select a subset of the data. E.g., if datasplit=3, then
    # randomly choose 1/3 of the data.
    if data_split > 1:
        indices = torch.tensor(np.arange(len(trainset)))
        data_num = len(trainset) // data_split  # the number of data in a chunk of the split

        # Randomly sample indices. Use seed=0 in the generator to make this reproducible
        state = np.random.get_state()
        np.random.seed(0)
        indices = np.random.choice(indices, data_num, replace=False)
        np.random.set_state(state)

        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   sampler=train_sampler,
                                                   shuffle=False, num_workers=threads)
    else:
        kwargs = {'num_workers': 2, 'pin_memory': True}
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=False, **kwargs)
    testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                           download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=threads)

    return train_loader, test_loader


def get_data_loaders_for_training(args):
    if args.trainloader and args.testloader:
        assert os.path.exists(args.trainloader), 'trainloader does not exist'
        assert os.path.exists(args.testloader), 'testloader does not exist'
        trainloader = torch.load(args.trainloader)
        testloader = torch.load(args.testloader)
        return trainloader, testloader

    normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

    if args.raw_data:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        if not args.noaug:
            # with data augmentation
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            # no data augmentation
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    kwargs = {'num_workers': 2, 'pin_memory': True} if args.ngpu else {}
    trainset = torchvision.datasets.CIFAR10(root=get_relative_path('data'), train=True, download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=get_relative_path('data'), train=False, download=True,
                                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                              shuffle=True, **kwargs)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, **kwargs)

    return trainloader, testloader
