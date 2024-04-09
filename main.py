from argparse import Namespace
from resnet import ResNet, BasicBlock
import os
from torch import torch
from collections import OrderedDict


from args import parse_args
from train import train
from test import test

if __name__ == '__main__':
    args: Namespace = parse_args()
    if args.pretrained is None:
        resnet: ResNet = train(args)
        
        path: str = os.path.join(args.save_dir, f'resnet_{args.csv_suffix}.pt')
        torch.save(resnet.state_dict(), path)
    else:
        path: str = os.path.join(args.save_dir, args.pretrained)
        state_dict: OrderedDict = torch.load(path)
        resnet: ResNet = ResNet(BasicBlock, [2, 1, 1, 1])
        resnet.load_state_dict(state_dict)

    if args.test_filename is not None:
        test(args, resnet)