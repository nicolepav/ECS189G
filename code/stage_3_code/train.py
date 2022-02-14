'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import CNN
from src.result_saver import Result_Saver
from src.evaluation import Evaluate_Accuracy
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import sys
import torch
import matplotlib.pyplot as plt


# ---- Convolutional Neural Network Script MODIFIED FOR MNIST----

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
    train_data_obj = Dataset_Loader('MNIST', 'Grayscale Handwritten Numbers', dataset_path)
    test_data_obj = Dataset_Loader('MNIST', 'Grayscale Handwritten Numbers', dataset_path)
    train_data_obj.load(train=True)
    test_data_obj.load(train=False)

    # MNIST
    # https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
    in_channels = 1  # 1 for grayscale, 3 for color RGB
    layers_data = [


        nn.Conv2d(1, 6, kernel_size=(5, 5), padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(6, 16, kernel_size=(5, 5)),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Linear(400, 120),
        nn.ReLU(),
        nn.Linear(120, 84),
        nn.ReLU(),
        # nn.Softmax(dim=1),
        nn.Linear(84, 10),
        nn.ReLU(),
        nn.Dropout(0.25)
        # nn.Softmax(dim=1)

    ]

    # Usage: CNN(layers_data, learning_rate, epoch, batch, optimizer, loss_function, device)
    method_obj = CNN(layers_data, 1e-3, 60, 100, torch.optim.Adam, nn.CrossEntropyLoss(), device)

    result_obj = Result_Saver('Model saver', '')
    result_obj.result_destination_file_path = 'result/CNN_model.pth'

    evaluate_obj = Evaluate_Accuracy('Accuracy', '')
    # ------------------------------------------------------

    # ---- Running Section ---------------------------------
    print('************ Start ************')

    # temp = torch.FloatTensor(train_data_obj.data['X'])
    # print(temp)

    X_train = torch.FloatTensor(train_data_obj.data['X']).reshape(60000, 1, 28, 28)/255
    y_train = torch.LongTensor(train_data_obj.data['y'])
    X_test = torch.FloatTensor(test_data_obj.data['X']).reshape(10000, 1, 28, 28)/255
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

