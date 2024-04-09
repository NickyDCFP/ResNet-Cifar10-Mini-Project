import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD
import pandas as pd
from tqdm import tqdm
from time import time

from constants import MAX_PARAMS
from resnet import ResNet, BasicBlock
from entropysgd import EntropySGD
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
    weight_decay: float = args.weight_decay
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device {device}")
    
    if args.optim == 'adam':
        opt: Adam = Adam(
            params=resnet.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif args.optim == 'adamw':
        opt: AdamW = AdamW(
            params=resnet.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif args.optim == 'entropysgd':
        opt: EntropySGD = EntropySGD(
            params=resnet.parameters(),
            config=dict(lr=lr, weight_decay=weight_decay)
        )
    elif args.optim == 'sgd':
        opt: SGD = SGD(
            params=resnet.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    
    loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=0.2)

    metrics: list[str] = ['Train Loss', 'Val Loss']
    history: pd.DataFrame = pd.DataFrame(columns=metrics)

    train_dataloader: DataLoader; val_dataloader: DataLoader
    train_dataloader, val_dataloader = get_dataset(args)

    resnet = resnet.to(device)

    for epoch in range(epochs):
        train_loss: float = 0.0
        val_loss: float = 0.0
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
            if args.optim == "entropysgd":
                def closure():
                    opt.zero_grad()
                    y_pred: torch.Tensor = resnet(x)
                    fit: torch.Tensor = loss(y_pred, y)
                    fit.backward()
                    return fit
                opt.step(closure)
            else:
                opt.step()
            train_loss += fit.item()
        train_loss /= len(train_dataloader)
        print(f'Train Loss: {train_loss}')
        print('Validating...')
        resnet.eval()
        for _, (x, y) in enumerate(tqdm(val_dataloader)):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)
                y_pred = resnet(x)
                fit = loss(y_pred, y)
                val_loss += fit.item()
        val_loss /= len(val_dataloader)
        history.loc[len(history)] = [train_loss, val_loss]
        time_elapsed: float = time() - start_time
        print(f'Val Loss: {val_loss}')
        print(f'Time Elapsed: {time_elapsed:.2f}s')
    history.to_csv(f"history_{args.csv_suffix}.csv")

    print('Validating...')
    correct_predictions: int = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(tqdm(val_dataloader)):
            x = x.to(device)
            y = y.to(device)
            out = resnet(x)
            y_pred = torch.argmax(out, dim=1)
            fit = loss(out, y)
            val_loss += fit.item()
            correct_predictions += (y_pred == y).sum().item()
    print(f"Final Validation Accuracy: {correct_predictions * 100 / len(val_dataloader.dataset)}")
    print(f"Final Validation Loss: {val_loss / len(val_dataloader)}")

    return resnet