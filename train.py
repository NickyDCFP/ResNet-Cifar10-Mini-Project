import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW, SGD, Optimizer
import pandas as pd
from argparse import Namespace
from typing import Iterator

from constants import MAX_PARAMS
from resnet import ResNet, BasicBlock
from data import get_dataset

def get_optim(
    name: str,
    params: Iterator[Parameter],
    lr: float,
    weight_decay: float
) -> Optimizer:
    """
    Grabs the optimizer of the specified name and initializes it.

    Parameters:
        name:           the name of the desired optimizer: adam, adamw, or sgd
        params:         the parameters of the model the optimizer will optimize
        lr:             the learning rate the model is to be trained with
        weight_decay:   the L2 regularization parameter for the model
    
    Returns:
        The optimizer to be trained with
    """
    if name == 'adam':
        return Adam(params=params, lr=lr, weight_decay=weight_decay)
    if name == 'adamw':
        return AdamW(params=params, lr=lr, weight_decay=weight_decay)
    if name == 'sgd':
        return SGD(params=params, lr=lr, weight_decay=weight_decay)


def validate(model: ResNet, val_dataloader: DataLoader, device: str) -> None:
    """
    Validates the model against the validation set. Prints the final validation
    accuracy and loss.

    Parameters:
        model:          The model to be validated against the validation set
        val_dataloader: The validation dataset in a DataLoader
        device:         The device the model is on, and on which it is to
                        be validated
    """
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

def train_one_epoch(
    model: ResNet,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    loss: nn.CrossEntropyLoss,
    opt: Optimizer,
    epoch: int,
    device: str,
    history: pd.DataFrame,
) -> None:
    """
    Trains the model for one epoch and records the training and validation loss.

    Parameters:
        model:              the model to be trained for one epoch
        train_dataloader:   the training data, loaded into a dataloader
        val_dataloader:     the validation data, loaded into a dataloader
        loss:               the loss function to be used to train the model
        opt:                the optimizer to be used to train the model
        epoch:              the number of the current epoch
        device:             the device the model is to be trained on
        history:            the history dataframe to track the model's training
    """
    train_loss: float = 0.0
    val_loss: float = 0.0
    x: torch.Tensor; y: torch.Tensor
    model.train()
    print(f"Epoch {epoch + 1}")
    for _, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad()
        y_pred: torch.Tensor = model(x)
        fit: torch.Tensor = loss(y_pred, y)
        fit.backward()
        opt.step()
        train_loss += fit.item()
    train_loss /= len(train_dataloader)
    model.eval()
    with torch.no_grad():
        for _, (x, y) in enumerate(val_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            fit = loss(y_pred, y)
            val_loss += fit.item()
    val_loss /= len(val_dataloader)
    history.loc[len(history)] = [train_loss, val_loss]

def train_model(
    args: Namespace,
    metrics: str,
    device: str,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> ResNet:
    """
    Trains the model on the training set and saves the history.

    Parameters:
        args:               arguments from the command line
        metrics:            the metrics to be tracked in the history
        device:             the device on which the model is to be trained
        train_dataloader:   the training data in a DataLoader
        val_dataloader:     the validation data in a DataLoader
    
    Returns:
        The trained model
    """
    model = ResNet(BasicBlock, [2, 1, 1, 1])
    opt: Optimizer = get_optim(
        args.optim,
        model.parameters(),
        args.lr,
        args.weight_decay
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params <= MAX_PARAMS, (
        f"Expected <= {MAX_PARAMS} parameters " \
        f"but model has {n_params} parameters."
    )
    model = model.to(device)
    history: pd.DataFrame = pd.DataFrame(columns=metrics)

    loss: nn.CrossEntropyLoss = nn.CrossEntropyLoss(label_smoothing=0.2)
    for epoch in range(args.epochs):
        train_one_epoch(
            model,
            train_dataloader,
            val_dataloader,
            loss,
            opt,
            epoch,
            device,
            history,
        )
    history.to_csv(f"history_{args.csv_suffix}.csv")

def train(args):
    """
    Loads in the dataset, trains the model, and prints the final validation
    accuracy.

    Parameters:
        args:       arguments from the commmand line
    
    Returns:
        The trained model
    """
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

    validate(model, val_dataloader, device)

    return model