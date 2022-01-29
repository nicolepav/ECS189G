''
TBD
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from src.dataset_loader import Dataset_Loader
from src.model import MLP
from src.result_saver import Result_Saver
from src.setting_test import Setting_Test
from src.evaluation import Evaluate_Accuracy
import numpy as np
import torch
from torch import nn
import sys


# ---- Multi-Layer Perceptron Script ----
if 1:

    if len(sys.argv) != 2:
        print("Usage: python test.py [pth_path]")
        exit()
    pth_path = sys.argv[1]

    # ---- Parameter Section -------------------------------
    use_cuda = torch.cuda.is_available()
    np.random.seed(48)
    torch.manual_seed(48)
    if use_cuda:
        torch.cuda.manual_seed(48)
    # ------------------------------------------------------

    # ---- Objection Initialization Section ---------------
    data_obj = Dataset_Loader('Testing Set',
                                   '10,000 lines: label {0, 1, 2, ..., 9}, feature_1, feature_2, ..., feature_784')
    data_obj.dataset_source_folder_path = 'code/stage_2_code/data/'
    data_obj.dataset_source_file_name = 'test.csv'

    layers_data = [
        (10, nn.ReLU),
        (10, nn.Softmax)
    ]
    method_obj = MLP(784, layers_data, 1e-3, torch.optim.Adam, nn.CrossEntropyLoss(), 500, [])
    method_obj.load_state_dict(torch.load(pth_path))

    result_obj = Result_Saver('Prediction saver', '')
    result_obj.result_destination_folder_path = 'code/stage_2_code/result/'
    result_obj.result_destination_file_name = 'MLP_prediction_result'

    setting_obj = Setting_Test('Testing', '')
    #setting_obj = Setting_KFold_CV()
    evaluate_obj = Evaluate_Accuracy('Accuracy', '')
    # ------------------------------------------------------

    # ---- Running Section ---------------------------------
    print('************ Start ************')
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj, use_cuda)
    setting_obj.print_setup_summary()
    score = setting_obj.load_run_save_evaluate()
    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(score))
    print('************ Finish ************')
    # ------------------------------------------------------
    print(method_obj.training_loss)
