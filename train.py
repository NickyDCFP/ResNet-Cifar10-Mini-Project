import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import pandas as pd
from resnet import ResNet, BasicBlock
from tqdm import tqdm
from time import time
from constants import MAX_PARAMS

from data import get_dataset

def train(args):
    resnet = ResNet(BasicBlock, [2, 1, 1, 1])
    n_params = sum(p.numel() for p in resnet.parameters())
    assert n_params <= MAX_PARAMS, (
        f"Expected <= {MAX_PARAMS} parameters " \
        f"but model has {n_params} parameters."
    )

    lr: float = args.lr
    epochs: int = args.epochs
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device {device}")

    opt: Adam = Adam(
        params=resnet.parameters(),
        lr=lr
    ) # try adamw
    loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=0.1)

    metrics: list[str] = ['Train Loss', 'Test Loss']
    history: pd.DataFrame = pd.DataFrame(columns=metrics)

    train_dataloader: DataLoader; test_dataloader: DataLoader
    train_dataloader, test_dataloader = get_dataset(args)

    resnet = resnet.to(device)


    for epoch in range(epochs):
        train_loss: float = 0.0
        test_loss: float = 0.0
        start_time: float = time()
        x: torch.Tensor; y: torch.Tensor
        print(f'Epoch {epoch + 1}')
        print('Training...')
        resnet.train()
        for _, (x, y) in enumerate(tqdm(train_dataloader)):
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            y_pred: torch.Tensor = resnet(x)
            fit: torch.Tensor = loss(y_pred, y)
            fit.backward()
            opt.step()
            train_loss += fit.item()
        train_loss /= len(train_dataloader)
        print(f'Train Loss: {train_loss}')
        print('Testing...')
        resnet.eval()
        for _, (x, y) in enumerate(tqdm(test_dataloader)):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                y_pred = resnet(x)
                fit = loss(y_pred, y)
                test_loss += fit.item()
        test_loss /= len(test_dataloader)
        history.loc[len(history)] = [train_loss, test_loss]
        time_elapsed: float = time() - start_time
        print(f'Test Loss: {test_loss}')
        print(f'Time Elapsed: {time_elapsed:.2f}s')

    return resnet