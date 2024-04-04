import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

def get_dataset(args) -> tuple[DataLoader, DataLoader]:
    train_transforms: v2.Compose = v2.Compose(
        [
            v2.RandomResizedCrop(size=32, antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
    )
    test_transforms: v2.Compose = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]
    )
    train_set: CIFAR10 = CIFAR10(
        args.data_dir,
        train=True,
        download=True,
        transform=train_transforms,
    )
    test_set: CIFAR10 = CIFAR10(
        args.data_dir,
        train=False,
        download=True,
        transform=test_transforms
    )

    train_dataloader: DataLoader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True
    )
    test_dataloader: DataLoader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False
    )
    return train_dataloader, test_dataloader