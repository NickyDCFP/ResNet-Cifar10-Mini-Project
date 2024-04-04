from argparse import Namespace
from resnet import ResNet

from args import parse_args
from train import train

if __name__ == '__main__':
    args: Namespace = parse_args()
    resnet: ResNet = train(args)