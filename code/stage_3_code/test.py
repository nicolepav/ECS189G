'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import CNN
from src.evaluation import Evaluate_Accuracy
import numpy as np
import sys
import torch
from torch import nn


#---- Convolutional Neural Network Script ----
if 1:
    if len(sys.argv) != 3:
        print("Usage: python test.py [dataset_path] [pth_path]")
        exit(1)
    dataset_path = sys.argv[1]
    pth_path = sys.argv[2]

    # ---- Parameter Section -------------------------------
    use_cuda = torch.cuda.is_available()
    np.random.seed(48)
    torch.manual_seed(48)
    if use_cuda:
        torch.cuda.manual_seed(48)
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    # ------------------------------------------------------

    # ---- Objection Initialization Section ---------------
    test_data_obj = Dataset_Loader('Testing Set', 'TBD', dataset_path)
    test_data_obj.load(train=False)

    # Default setting for CIFAR dataset
    in_channels = 3  # 1 for grayscale, 3 for color RGB
    num_classes = 10
    layers_data = [  # Baseline: VGG16
        # Layer 1
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 2
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 3
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 4
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 5
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 6
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 7
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 8
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 9
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 10
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 11
        nn.Conv2d(256, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 12
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 13
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 14
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 15
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 16
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 17
        nn.Conv2d(512, 512, kernel_size=3, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # Layer 18
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.AvgPool2d(kernel_size=1, stride=1),

        # fc Layer (dense layer)
        # Before the dense layer, need to flatten all dimensions except batch
        # This is down in the model.py (keep in mind)
        nn.Linear(512, num_classes)
    ]

    # Usage: CNN(layers_data, learning_rate, epoch, batch, optimizer, loss_function, device)
    method_obj = CNN(layers_data, 1e-3, 30, 64, torch.optim.Adam, nn.CrossEntropyLoss(), device)
    method_obj.load_state_dict(torch.load(pth_path))

    evaluate_obj = Evaluate_Accuracy('Accuracy', '')

    # ---- Running Section ---------------------------------
    print('************ Start ************')
    X_test = torch.FloatTensor(test_data_obj.data['X']).permute(0, 3, 1, 2)
    y_test = torch.LongTensor(test_data_obj.data['y'])

    # GPU
    if use_cuda:
        method_obj.cuda()

    # Run CNN model
    method_obj.data = {'test': {'X': X_test, 'y': y_test}}
    testing_result = method_obj.run()

    # Evaluate predication result
    evaluate_obj.data = testing_result
    score = evaluate_obj.evaluate()

    print('************ Overall Performance ************')
    print('CNN Accuracy: ' + str(score))
    print('************ Finish ************')
    # ------------------------------------------------------


