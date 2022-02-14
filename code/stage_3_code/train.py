'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from sklearn import datasets
from src.dataset_loader import Dataset_Loader
from src.model import CNN
from src.result_saver import Result_Saver
from src.evaluation import Evaluate_Accuracy
import numpy as np
import sys
import torch
from torch import nn

# import pickle
# import pandas as pd

# # ---- View dataset ------------------------------
# if 1:
#     f = open('datasets/stage_3_data/CIFAR', 'rb')
    
#     data = pickle.load(f)
#     f.close()

#     print('training set size:', len(data['train']), 'testing set size:', len(data['test']))
#     for pair in data['train']:
#    # for pair in data['test']:
#        plt.imshow(pair['image'], cmap="Greys")
#        plt.show()
#        print(pair['label'])


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
    in_channels = 3             # 1 for grayscale, 3 for color RGB
    num_classes = 10
    layers_data = [             # Implements Lenet-5
        # Layer 1       32 * 32 * 3
        nn.Conv2d(in_channels, out_channels=6, kernel_size=5, stride=1, padding=0, dilation=1),  # (32 - 5) / 1 + 1 = 28
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (28 - 2) / 2 + 1 = 14

        # Layer 2       14 * 14 * 6
        nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, dilation=1),           # (14 - 5) / 1 + 1 = 10
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),                           # (10 - 2) / 2 + 1 = 5

        # Layer 3       5 * 5 * 16
        nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0, dilation=1),          # (5 - 5) / 1 + 1 = 1
        nn.ReLU(),

        # fc Layer (dense layer)
        # Before the dense layer, need to flatten all dimensions except batch
        # This is down in the model.py (keep in mind)
        nn.Linear(120, 84),
        nn.ReLU(),
        nn.Linear(84, 10)
    ]

    # Usage: CNN(layers_data, learning_rate, epoch, batch, optimizer, loss_function, device)
    method_obj = CNN(layers_data, 1e-3, 30, 256, torch.optim.Adam, nn.CrossEntropyLoss(), device)

    result_obj = Result_Saver('Model saver', '')
    result_obj.result_destination_file_path = 'result/CNN_model.pth'

    evaluate_obj = Evaluate_Accuracy('Accuracy', '')
    # ------------------------------------------------------

    # ---- Running Section ---------------------------------
    print('************ Start ************')
    X_train = torch.FloatTensor(train_data_obj.data['X']).reshape(-1, 3, 32, 32)
    y_train = torch.LongTensor(train_data_obj.data['y'])
    X_test = torch.FloatTensor(test_data_obj.data['X']).reshape(-1, 3, 32, 32)
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

