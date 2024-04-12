from argparse import ArgumentParser, Namespace

def parse_args() -> Namespace:
    parser = ArgumentParser("ResNet training on CIFAR10 dataset")

    parser.add_argument(
        "--data-dir",
        help="The directory in which the dataset should be stored",
        type=str,
        default="./dataset/"
    )
    parser.add_argument(
        "--save-dir",
        help="The directory in which the trained models should be stored",
        type=str,
        default="./trained_models/"
    )
    parser.add_argument(
        "--batch-size",
        help="The batch size for the model",
        type=int,
        default=128
    )
    parser.add_argument(
        "--lr",
        help="The learning rate for the model",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--epochs",
        help="The number of epochs to train the model",
        type=int,
        default=350
    )
    parser.add_argument(
        "--weight-decay",
        help="The L2 regularization parameter for the model",
        type=float,
        default=0.001
    )
    parser.add_argument(
        "--optim",
        help="The optimizer to be used for training the model",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adamw"
    )
    parser.add_argument(
        "--pretrained",
        help="The pretrained model, if any, to test on the test data",
        type=str,
        default=None
    )
    parser.add_argument(
        "--test-filename",
        help="The filename for the test data from Kaggle",
        type=str,
        default="cifar_test_nolabels.pkl"
    )
    parser.add_argument(
        "--csv-suffix",
        help="The suffix for the .csv files so they don't overwrite each other",
        type=str,
        default=""
    )

    args = parser.parse_args()
    return args