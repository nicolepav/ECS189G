'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import LSTM, train, test
from src.result_saver import Result_Saver
from src.evaluation import Evaluate_Accuracy

import numpy as np
import sys
import torch
import matplotlib.pyplot as plt
from torch import nn
from torchtext.legacy.data import BucketIterator
from os.path import exists


#---- Long Short-Term Memory Script ----
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
    print('************ Initialization ************')
    data_obj = Dataset_Loader('Training and Testing Set', 'Text Classification', dataset_path)
    if exists(dataset_path + '\\train.csv') and exists(dataset_path + '\\test.csv'):
        print('Skip: train.csv and test.csv are found at ' + dataset_path)
    else:
        data_obj.txt2csv()
        print('train.csv and test.csv are now created at ' + dataset_path)
    data_obj.load(classification=True)
    #print(vars(data_obj.train_data.examples[0]))

    # Default setting
    vocab_size = len(data_obj.text_field.vocab)
    n_layers = 3
    embedding_dim = 100
    hidden_dim = 256
    output_dim = 1
    epoch = 5
    batch_size = 64
    learning_rate = 0.001
    loss_function = nn.BCELoss()
    optimizer = torch.optim.Adam

    # Usage: CNN(layers_data, learning_rate, epoch, batch, optimizer, loss_function, device)
    method_obj = LSTM(vocab_size, n_layers, embedding_dim, hidden_dim, output_dim, epoch, learning_rate, loss_function, optimizer, device)
    method_obj = method_obj.to(device)

    # Load the pretrained vocab vector to the embedding layer
    # Reset the vocab vectors of unk and pad to 0
    method_obj.embedding.weight.data.copy_(data_obj.text_field.vocab.vectors)
    method_obj.embedding.weight.data[data_obj.text_field.vocab.stoi[data_obj.text_field.unk_token]] = torch.zeros(embedding_dim)
    method_obj.embedding.weight.data[data_obj.text_field.vocab.stoi[data_obj.text_field.pad_token]] = torch.zeros(embedding_dim)

    result_obj = Result_Saver('Model saver', '')
    result_obj.result_destination_file_path = 'result/LSTM_model'

    evaluate_obj = Evaluate_Accuracy('Accuracy', '')
    # ------------------------------------------------------

    # ---- Running Section ---------------------------------
    print('************ Start ************')

    # LSTM iterator
    train_iter, test_iter = BucketIterator.splits((data_obj.train_data, data_obj.test_data),
                                                  batch_size=batch_size,
                                                  sort_within_batch=True,
                                                  sort_key=lambda x: len(x.text),
                                                  device=device)

    # Run LSTM model
    train(method_obj, train_iter)
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(method_obj.training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    score = test(method_obj, test_iter)

    # Save learned model's parameters
    result_obj.data = method_obj.state_dict()
    result_obj.save_learned_model()

    print('************ Overall Performance ************')
    print('LSTM Accuracy: ' + str(score))
    print('************ Finish ************')
    # ------------------------------------------------------

