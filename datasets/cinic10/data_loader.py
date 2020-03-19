"""
Load a dataset of images by specifying the folder where its located.
"""


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
from multiprocessing import Pool
import numpy as np

# Torch related stuff
import torch.utils.data as data
import torchvision
from torchvision.datasets.folder import pil_loader, ImageFolder
import torchvision
import torchvision.transforms as transforms



def get_data_loaders(args=None):
    """
    See: https://github.com/BayesWatch/cinic-10
    """
    cinic_directory = 'datasets/cinic10/data'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]
    testloader = data.DataLoader(
        torchvision.datasets.ImageFolder(cinic_directory + '/test',
            transform=transforms.Compose([transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean, std=cinic_std)])),
        batch_size=128, shuffle=True)
    trainloader = testloader

    return trainloader, testloader

def ImageFolderDataset(path, inmem, workers, **kwargs):
    """Return the choosen dataset depending on the inmeme parameter
    Parameters
    ----------
    path : string
        Path to the dataset on the file System
    inmem : boolean
        Load the whole dataset in memory. If False, only file names are stored and images are loaded
        on demand. This is slower than storing everything in memory.
    workers: int
        Number of workers to use for the dataloaders
    Returns
    -------
    torch.utils.data.Dataset
        Split at the chosen path
    """
    return ImageFolderInMemory(path, workers) if inmem else ImageFolderTorchVision(path)


class ImageFolderTorchVision(ImageFolder):
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample, target = self.transform(sample, target)

        return sample, {'category_id': target}


class ImageFolderInMemory(data.Dataset):
    """
    This class loads the data provided and stores it entirely in memory as a dataset.
    It makes use of torchvision.datasets.ImageFolder() to create a dataset. Afterward all images are
    sequentially stored in memory for faster use when paired with dataloders. It is responsibility of
    the user ensuring that the dataset actually fits in memory.
    """

    def __init__(self, path, transform=None, target_transform=None, workers=1):
        """
        Load the data in memory and prepares it as a dataset.
        Parameters
        ----------
        path : string
            Path to the dataset on the file System
        transform : torchvision.transforms
            Transformation to apply on the data
        target_transform : torchvision.transforms
            Transformation to apply on the labels
        workers: int
            Number of workers to use for the dataloaders
        """
        self.dataset_folder = os.path.expanduser(path)
        self.transform = transform
        self.target_transform = target_transform

        # Get an online dataset
        dataset = torchvision.datasets.ImageFolder(path)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Load all samples
        pool = Pool(workers)
        self.data = pool.map(pil_loader, file_names)
        pool.close()

        # Set expected class attributes
        self.classes = np.unique(self.labels)

    def __getitem__(self, index):
        """
        Retrieve a sample by index
        Parameters
        ----------
        index : int
        Returns
        -------
        img : FloatTensor
        target : dict
            label of the image
        """

        img, target = self.data[index], self.labels[index]
        if self.transform is not None:
            img, target = self.transform(img, target)
        return img, {'category_id': target}

    def __len__(self):
        return len(self.data)
