'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import CNN
from src.result_saver import Result_Saver
from src.evaluation import Evaluate_Accuracy
import numpy as np
import sys
import torch
from torch import nn


#---- Convolutional Neural Network Script ----
if 1:
    if len(sys.argv) != 2:
        print("Usage: python train.py [dataset_path]")
        exit(1)
    dataset_path = sys.argv[1]

    # ---- Parameter Section -------------------------------
    use_cuda = torch.cuda.is_available()
    np.random.seed(48)
    torch.manual_seed(48)
    if use_cuda:
        torch.cuda.manual_seed(48)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # ------------------------------------------------------

    # ---- Objection Initialization Section ---------------
    train_data_obj = Dataset_Loader('Training Set', 'TBD', dataset_path)
    test_data_obj = Dataset_Loader('Testing Set', 'TBD', dataset_path)
    train_data_obj.load(train=True)
    test_data_obj.load(train=False)

    # Default setting for CIFAR dataset
    in_channels = 1             # 1 for grayscale, 3 for color RGB
    num_classes = 10
    layers_data = [             # Baseline: AlexNet
        # Layer 1       32 * 32 * 3
        nn.Conv2d(in_channels, 6, kernel_size=3, stride=1, padding=1, dilation=1),  # (32 + 2 - 3) / 1 + 1 = 32
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (32 - 2) / 2 + 1 = 16

        # Layer 2       16 * 16 * 6
        nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1, dilation=1),           # (16 - 3 + 2) / 1 + 1 = 16
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (16 - 2) / 2 + 1 = 8

        # Layer 3       8 * 8 * 16
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, dilation=1),          # (8 - 3 + 2) / 1 + 1 = 8
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (8 - 2) / 2 + 1 = 4

        # Layer 4       4 * 4 * 64
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, dilation=1),          # (4 - 3 + 2) / 1 + 1 = 4
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (4 - 2) / 2 + 1 = 2

        # Layer 5       2 * 2 * 128
        #nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, dilation=1),         # (2 - 3 + 2) / 1 + 1 = 2
        #nn.ReLU(),
        #nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (2 - 2) / 2 + 1 = 1

        # fc Layer (dense layer)
        # Before the dense layer, need to flatten all dimensions except batch
        # This is down in the model.py (keep in mind)
        nn.Linear(64, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, num_classes)
    ]

    # Usage: CNN(layers_data, learning_rate, epoch, batch, optimizer, loss_function, device)
    method_obj = CNN(layers_data, 1e-3, 10, 64, torch.optim.Adam, nn.CrossEntropyLoss(), device)

    result_obj = Result_Saver('Model saver', '')
    result_obj.result_destination_file_path = 'result/CNN_model'

    evaluate_obj = Evaluate_Accuracy('Accuracy', '')
    # ------------------------------------------------------

    # ---- Running Section ---------------------------------
    print('************ Start ************')
    X_train = torch.FloatTensor(train_data_obj.data['X']).reshape((-1, 1, 28, 28))
    y_train = torch.LongTensor(train_data_obj.data['y'])
    X_test = torch.FloatTensor(test_data_obj.data['X']).reshape((-1, 1, 28, 28))
    y_test = torch.LongTensor(test_data_obj.data['y'])

    # GPU
    if use_cuda:
        method_obj.cuda()

    # Run CNN model
    method_obj.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
    learned_result = method_obj.run()

    # Save learned model's parameters
    result_obj.data = method_obj.state_dict()
    result_obj.save_learned_model()

    # Evaluate predication result
    evaluate_obj.data = learned_result
    score = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(score))
    print('************ Finish ************')
    # ------------------------------------------------------

