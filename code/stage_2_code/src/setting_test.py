'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from .base.setting import setting
import torch


class Setting_Test(setting):

    def load_run_save_evaluate(self):
        # Load dataset
        loaded_data = self.dataset.load()
        X_test = torch.FloatTensor(loaded_data['X'])
        y_test = torch.LongTensor(loaded_data['y'])

        # GPU
        if self.use_cuda:
            self.method.cuda()
            X_test = X_test.cuda()
            y_test = y_test.cuda()

        self.method.data = {'test': {'X': X_test, 'y': y_test}}
        # self.method.eval() if having dropout and batch normalization layers
        pred_result = self.method.test(X_test)
        test_result = {'pred_y': pred_result, 'true_y': y_test}

        # Save predication result
        self.result.data = test_result
        self.result.save_learned_result()

        # Evaluate predication result
        self.evaluate.data = test_result
        return self.evaluate.evaluate()

