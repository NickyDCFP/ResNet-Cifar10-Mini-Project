from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser = ArgumentParser("ResNet training on CIFAR10 dataset")

    parser.add_argument("--data-dir", type=str, default="./dataset/")
    parser.add_argument("--save-dir", type=str, default="./trained_models/")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--optim", type=str, choices=["adam", "adamw", "entropysgd", "sgd"], default="adam")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--test-filename", type=str, default="cifar_test_nolabels.pkl")
    parser.add_argument("--csv-suffix", type=str, default="")

    args = parser.parse_args()
    return args