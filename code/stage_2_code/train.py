'''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import MLP
from src.result_saver import Result_Saver
from src.setting_kfold_cv import Setting_KFold_CV
from src.setting_train_test_split import Setting_Train_Test_Split
from src.evaluation import Evaluate_Accuracy
import numpy as np
import torch
from torch import nn


#---- Multi-Layer Perceptron Script ----
if 1:
    #---- Parameter Section -------------------------------
    use_cuda = torch.cuda.is_available()
    np.random.seed(48)
    torch.manual_seed(48)
    if use_cuda:
        torch.cuda.manual_seed(48)
    #------------------------------------------------------

    # ---- Objection Initialization Section ---------------
    data_obj = Dataset_Loader('Training Set',
                                    '60,000 lines: label {0, 1, 2, ..., 9}, feature_1, feature_2, ..., feature_784')
    data_obj.dataset_source_folder_path = 'data/'
    data_obj.dataset_source_file_name = 'train.csv'

    layers_data = [
        (10, nn.ReLU),
        (10, nn.Softmax)
    ]

    # Usage: MLP(input_size, layers_data, learning_rate, optimizer, loss_function, epoch)
    method_obj = MLP(784, layers_data, 1e-3, torch.optim.Adam, nn.CrossEntropyLoss(), 500)

    result_obj = Result_Saver('Model saver', '')
    result_obj.result_destination_folder_path = 'result/'
    result_obj.result_destination_file_name = 'MLP_model'

    setting_obj = Setting_KFold_CV('K Fold Cross Validation', '')
    # Not supported: setting_obj = Setting_Train_Test_Split('Train Test Split', '')

    evaluate_obj = Evaluate_Accuracy('Accuracy', '')
    # ------------------------------------------------------

    # ---- Running Section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj, use_cuda)
    setting_obj.print_setup_summary()
    mean_score, std_score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score) + ' +/- ' + str(std_score))
    print('************ Finish ************')
    # ------------------------------------------------------

