'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.setting import setting
from sklearn.model_selection import KFold
import numpy as np
import torch

class Setting_KFold_CV(setting):
    fold = 3
    
    def load_run_save_evaluate(self):
        # Load dataset
        loaded_data = self.dataset.load()
        
        kf = KFold(n_splits=self.fold, shuffle=True)
        
        fold_count = 0
        score_list = []
        for train_index, test_index in kf.split(loaded_data['X']):
            fold_count += 1
            print('************ Fold:', fold_count, '************')
            X_train, X_test = torch.FloatTensor(loaded_data['X'])[train_index], torch.FloatTensor(loaded_data['X'])[test_index]
            y_train, y_test = torch.LongTensor(loaded_data['y'])[train_index], torch.LongTensor(loaded_data['y'])[test_index]

            # GPU
            if self.use_cuda:
                self.method.cuda()
                X_train = X_train.cuda()
                X_test = X_test.cuda()
                y_train = y_train.cuda()
                y_test = y_test.cuda()

            # Run MLP model
            self.method.data = {'train': {'X': X_train, 'y': y_train}, 'test': {'X': X_test, 'y': y_test}}
            learned_result = self.method.run()
            
            # Save learned model's parameters
            self.result.data = self.method.state_dict()
            self.result.fold_count = fold_count
            self.result.save_learned_model()

            # Evaluate predication result
            self.evaluate.data = learned_result
            score_list.append(self.evaluate.evaluate())
        
        return np.mean(score_list), np.std(score_list)

