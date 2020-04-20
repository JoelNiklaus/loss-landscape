"""
TODO Apply test transform from saved model

Use Test transform like this
    if 'test_transform' in checkpoint:
        self.transform = checkpoint['test_transform']
# Transform it
        img = self.transform(img)

"""

# Utils

# Torch related stuff
import os
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms

def get_relative_path(file):
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    return os.path.join(script_dir, file)

def get_data_loaders_for_plotting(datapath='data', batch_size=128,
                                  threads=2, raw_data=False, data_split=1):
    """
    See: https://github.com/BayesWatch/cinic-10
    """
    data_folder = get_relative_path(datapath)
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    test_loader = data.DataLoader(
        torchvision.datasets.ImageFolder(data_folder + '/test',
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize(mean=cinic_mean,
                                                                                            std=cinic_std)])),
        batch_size=batch_size, shuffle=True)
    train_loader = test_loader

    return train_loader, test_loader


def get_data_loaders_for_training(args):
    data_folder = get_relative_path('data')
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    test_loader = data.DataLoader(
        torchvision.datasets.ImageFolder(data_folder + '/test',
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize(mean=cinic_mean,
                                                                                            std=cinic_std)])),
        batch_size=args.batch_size, shuffle=True)
    train_loader = data.DataLoader(
        torchvision.datasets.ImageFolder(data_folder + '/train',
                                         transform=transforms.Compose([transforms.ToTensor(),
                                                                       transforms.Normalize(mean=cinic_mean,
                                                                                            std=cinic_std)])),
        batch_size=args.batch_size, shuffle=True)

    return train_loader, test_loader
