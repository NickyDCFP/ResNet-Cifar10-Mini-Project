from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser = ArgumentParser("ResNet training on CIFAR10 dataset")

    parser.add_argument("--data-dir", type=str, default="./dataset/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=500)

    args = parser.parse_args()
    return args