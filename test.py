import torch
from argparse import Namespace
from resnet import ResNet
import pandas as pd
from tqdm import tqdm

from data import get_test_data

def test(args: Namespace, resnet: ResNet) -> None:
    test_data: torch.Tensor; ids: torch.Tensor
    test_data, ids = get_test_data(args)
    
    df: pd.DataFrame = pd.DataFrame(columns=["ID", "Labels"])
    resnet.eval()
    resnet = resnet.cuda()
    test_data = test_data.cuda()
    for i in tqdm(range(test_data.shape[0])):
        ex = test_data[i:i+1, :, :, :]
        out: int = resnet(ex)
        y_pred: int = int(torch.argmax(out, dim=1).item())
        df.loc[len(df)] = [int(ids[i].item()), y_pred]
    df.to_csv(f'out_{args.csv_suffix}.csv', index=False)