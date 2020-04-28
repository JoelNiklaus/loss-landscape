"""
Load a dataset of images by specifying the folder where its located.
"""

# Utils
import os
from multiprocessing import Pool
import numpy as np

# Torch related stuff
import torch.utils.data as data
import torchvision
from torchvision.datasets.folder import pil_loader, ImageFolder

class temporary_seed:
    """
    Python context manager class that allows for consistent numpy random number generation.
    Adapted from https://stackoverflow.com/questions/49555991/can-i-create-a-local-numpy-random-seed
    """
    def __init__(self, seed):
        self.seed = seed
        self.backup = None

    def __enter__(self):
        self.backup = np.random.get_state()
        np.random.seed(self.seed)

    def __exit__(self, *_):
        np.random.set_state(self.backup)

class HydraImageFolder(data.Dataset):
    """
    This class loads the data provided and stores it entirely in memory as a dataset.

    It makes use of torchvision.datasets.ImageFolder() to create a dataset. Afterward all images are
    sequentially stored in memory for faster use when paired with dataloders. It is responsibility of
    the user ensuring that the dataset actually fits in memory.
    """

    def __init__(self, path, transform=None, target_transform=None, workers=1, in_memory=False,
                 heads=1, hydra_classes=10):
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
        self.inmem = in_memory
        self.heads = heads
        self.hydra_classes = hydra_classes

        # Set seed val
        self.seed_val = 0

        # Get an online dataset
        dataset = torchvision.datasets.ImageFolder(path)

        # Shuffle the data once (otherwise you get clusters of samples of same class in each minibatch for val and test)
        with temporary_seed(42):
            np.random.shuffle(dataset.imgs)

        # Extract the actual file names and labels as entries
        self.file_names = np.asarray([item[0] for item in dataset.imgs])
        self.labels = np.asarray([item[1] for item in dataset.imgs])

        # Generate fake labels for each file
        self.fake_labels = self._generate_fake_labels(self.file_names, self.heads, self.hydra_classes)

        # Load all samples
        if self.inmem == True:
            with Pool(workers) as pool:
                self.data = pool.map(pil_loader, self.file_names)

        # Set expected class attributes
        self.classes = np.unique(self.fake_labels)

    def regenerate_fake_labels(self):
        self.fake_labels = self._generate_fake_labels(self.file_names, self.heads, self.hydra_classes)
        return

    def __getitem__(self, index):
        """
        Retrieve a sample by index

        Parameters
        ----------
        index : int

        Returns
        -------
        img : FloatTensor
        target : int
            label of the image
        """
        if self.inmem == True:
            img = self.data[index]
        else:
            img = pil_loader(self.file_names[index])

        fake_target = self.fake_labels[:, index].item()

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            fake_target = self.target_transform(fake_target)
        return img, fake_target


    def _generate_fake_labels(self, file_names, heads, hydra_classes):
        with temporary_seed(self.seed_val):
            fake_labels = np.random.randint(0, hydra_classes, size=(heads, len(file_names)))
        self.seed_val += 1
        return fake_labels

    def __len__(self):
        return len(self.file_names)

