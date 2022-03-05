'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import GCN, train, test
from src.result_saver import Result_Saver

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as fun
from os.path import exists


# ---- Graph Convolutional Networks Script -----------------
if 1:
    # ---- Command Line Arguments Section ------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True,
                        help='The dataset\'s name specified for use.')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='The dataset\'s path specified for use.')
    parser.add_argument('--use_cpu', action='store_true', default=False,
                        help='Disables CUDA training and uses CPU only.')
    parser.add_argument('--fast_mode', action='store_true', default=False,
                        help='Disables validation during training pass.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for torch.')

    # Command line arguments parser
    args = parser.parse_args()
    if args.dataset_path[-1] != '\\':
        args.dataset_path += '/'
    args.dataset_path = args.dataset_path.replace('\\', '/')

    args.use_cuda = not args.use_cpu and torch.cuda.is_available()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    # ------------------------------------------------------

    # ---- Objection Initialization Section ----------------
    print('************ Initialization ************')
    # Loading dataset with corresponding node and link files
    data_obj = Dataset_Loader(args.dataset_name, 'Dataset contains two files: node and link', args.dataset_path)
    if exists(args.dataset_path + 'node') and exists(args.dataset_path + 'link'):
        print('node and link files are found at ' + args.dataset_path)
    else:
        print("Error: no node and link files specified for use")
        print("Example: python train.py --dataset_path dataset\\cora\\")
        print("Assuming the cora folder contains node and link files: dataset\\cora\\node and dataset\\cora\\link")
        exit(1)
    data_obj.load()

    # ---- Parameter Section -------------------------------
    n_feat = data_obj.data['graph']['X'].shape[1]
    n_hid = 16
    n_class = data_obj.data['graph']['y'].max().item() + 1
    dropout = 0.5
    epoch = 200
    loss_function = fun.nll_loss
    optimizer = torch.optim.Adam
    learning_rate = 0.01
    weight_decay = 5e-4
    # ------------------------------------------------------

    # ---- Training Section --------------------------------
    # Usage: GCN(n_feat, n_hid, n_class, dropout, epoch, loss_function, optimizer, learning_rate, weight_decay)
    model = GCN(n_feat, n_hid, n_class, dropout, epoch, loss_function, optimizer, learning_rate, weight_decay)

    # Put mode and dataset into GPU if CUDA is selected and available
    if args.use_cuda:
        model.cuda()
        data_obj.data['graph']['X'] = data_obj.data['graph']['X'].cuda()
        data_obj.data['graph']['utility']['A'] = data_obj.data['graph']['utility']['A'].cuda()
        data_obj.data['graph']['y'] = data_obj.data['graph']['y'].cuda()
        data_obj.data['train_test_val']['idx_train'] = data_obj.data['train_test_val']['idx_train'].cuda()
        data_obj.data['train_test_val']['idx_test'] = data_obj.data['train_test_val']['idx_test'].cuda()
        data_obj.data['train_test_val']['idx_val'] = data_obj.data['train_test_val']['idx_val'].cuda()

    print('************ Training Start ************')
    train(model, data_obj, args.fast_mode)

    # Save learned model's parameters
    result_obj = Result_Saver('GCN Model Saver', 'Model Parameters')
    result_obj.result_destination_file_path = 'result/GCN_model'
    result_obj.data = model.state_dict()
    result_obj.save_learned_model()

    # Training loss plot
    plt.figure(figsize=(10, 5))
    plt.title("Training Loss")
    plt.plot(model.training_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()

    print('************ Training End **************')
    # ------------------------------------------------------

    # ---- Testing Section ---------------------------------
    print('************ Testing Start *************')
    test_acc = test(model, data_obj)
    print('************ Overall Performance *******')
    print('GCN Accuracy: ' + str(test_acc))
    print('************ Finish ********************')
    # ------------------------------------------------------

