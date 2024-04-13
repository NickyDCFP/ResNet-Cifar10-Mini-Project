import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
import pickle
import os
from argparse import Namespace

def get_dataset(args: Namespace) -> tuple[DataLoader, DataLoader]:
    """
    Retrieves CIFAR10 with a set of transforms applied to it.

    Parameters:
        args:       arguments from the command line

    Returns
        The desired dataset, loaded into separate train and test DataLoaders.
    """
    train_transforms: v2.Compose = v2.Compose(
        [
            v2.RandomResizedCrop(size=32, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomGrayscale(p=0.4),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    val_transforms: v2.Compose = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_set: CIFAR10 = CIFAR10(
        args.data_dir,
        train=True,
        download=True,
        transform=train_transforms,
    )
    val_set: CIFAR10 = CIFAR10(
        args.data_dir,
        train=False,
        download=True,
        transform=val_transforms,
    )
    train_dataloader: DataLoader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True
    )
    val_dataloader: DataLoader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    return train_dataloader, val_dataloader

def get_test_data(args: Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Loads in the test dataset for Kaggle leaderboard placement.

    Parameters:
        args:   arguments from the command line
    
    Returns:
        The test data images and their respective IDs.
    """
    path: str = os.path.join(args.data_dir, args.test_filename)
    with open(path, 'rb') as fo:
        data: dict = pickle.load(fo, encoding='bytes')
    test_data: torch.Tensor = torch.tensor(data[b'data'], dtype=torch.float32)
    test_data = test_data.reshape(test_data.size(0), 3, 32, 32)
    test_ids: torch.Tensor = torch.Tensor(data[b'ids'])
    test_data /= 255
    transform: v2.Normalize = v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_data = transform(test_data)
    return test_data, test_ids