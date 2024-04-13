# ResNet-Cifar10-Mini-Project

Mini-Project for CS-GY 6953 Deep Learning with Professor Chinmay Hegde. Uses ResNet code from [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar).

All of the requirements for the environment are listed in `environment.yaml`. 

To run inference on the test data using our pretrained model, be sure you have `cifar_test_nolabels.pkl` in `./dataset/`. If you want to run inference on a different `.pkl` file, you can configure that with `--test-filename`, and if you want to change the directory of the `.pkl` file, you can do so with `--data-dir`. Next, run the following command to generate an `out_.csv` with the predictions:
```
python main.py --pretrained resnet.pt
```

To train and validate a model (and run that model on the test data), be sure you satisfy the above requirements and run the following command:
```
python main.py
```
This command will generate an `out_.csv` with predictions, a `history_.csv` with training history, and a `resnet_.pt` model checkpoint. The suffixes for these can be changed with `--csv-suffix` (ie, `resnet_{csv_suffix}.pt`).