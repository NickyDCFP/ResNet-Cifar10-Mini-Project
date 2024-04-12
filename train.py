import os
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD, Optimizer
import pandas as pd
from collections import OrderedDict
from argparse import Namespace
from typing import Iterator

from constants import MAX_PARAMS
from resnet import ResNet, BasicBlock, ResNet101
from data import get_dataset

def train(args):
    train_dataloader: DataLoader; val_dataloader: DataLoader
    train_dataloader, val_dataloader = get_dataset(args)
    metrics: list[str] = ['Train Loss', 'Val Loss']
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device {device}")

    model: ResNet = train_model(
        args,
        metrics,
        device,
        train_dataloader,
        val_dataloader
    )

    loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    print('Validating...')
    correct_predictions: int = 0
    val_loss: float = 0
    with torch.no_grad():
        for _, (x, y) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            y_pred = torch.argmax(out, dim=1)
            fit = loss(out, y)
            val_loss += fit.item()
            correct_predictions += (y_pred == y).sum().item()
    print(f"Final Validation Accuracy: {correct_predictions * 100 / len(val_dataloader.dataset)}")
    print(f"Final Validation Loss: {val_loss / len(val_dataloader)}")

    return model

def train_model(
    args: Namespace,
    metrics: str,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> ResNet:
    model = ResNet(BasicBlock, [2, 1, 1, 1])
    opt: Optimizer = get_optim(
        args.optim,
        model.parameters(),
        args.model_lr,
        args.model_weight_decay
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= MAX_PARAMS, (
        f"Expected <= {MAX_PARAMS} parameters " \
        f"but model has {n_params} parameters."
    )
    model = model.to(device)

    model_loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    for epoch in range(args.model_epochs):
        train_one_epoch(
            model,
            train_dataloader,
            val_dataloader,
            model_loss,
            opt,
            epoch,
            device,
            model_history,
            args,
            teacher
        )
    model_history: pd.DataFrame = pd.DataFrame(columns=metrics)
    model_history.to_csv(f"model_history_{args.csv_suffix}.csv")
    
def get_optim(
    name: str,
    params: Iterator[Parameter],
    lr: float,
    weight_decay: float
) -> Optimizer:
    if name == 'adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params=params, lr=lr, weight_decay=weight_decay)

def train_one_epoch(
    model: ResNet,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss: nn.CrossEntropyLoss,
    opt: Optimizer,
    epoch: int,
    device: str,
    history: pd.DataFrame,
    args: Namespace,
    teacher: ResNet = None,
) -> None:
    train_loss: float = 0.0
    val_loss: float = 0.0
    hard_targets: bool = teacher is None or \
                         (args.epochs - epoch <= args.model_hard_epochs)
    x: torch.Tensor; y: torch.Tensor
    model.train()
    if teacher is None:
        print(f"Teacher epoch {epoch + 1}")
    else:
        teacher.eval()
        print(f"Student epoch {epoch + 1}")
    for _, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        if hard_targets:
            y = y.to(device)
        else:
            with torch.no_grad():
                y = teacher(x)
        opt.zero_grad()
        y_pred: torch.Tensor = model(x)
        fit: torch.Tensor = loss(y_pred, y)
        fit.backward()
        opt.step()
        train_loss += fit.item()
    train_loss /= len(train_dataloader)
    model.eval()
    for _, (x, y) in enumerate(val_dataloader):
        with torch.no_grad():
            x = x.to(device)
            if hard_targets:
                y = y.to(device)
            else:
                with torch.no_grad():
                    y = teacher(x)
            y_pred = model(x)
            fit = loss(y_pred, y)
            val_loss += fit.item()
    val_loss /= len(val_dataloader)
    history.loc[len(history)] = [train_loss, val_loss]