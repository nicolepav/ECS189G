'''
Base SettingModule class for all experiment settings
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

import abc

#-----------------------------------------------------
class setting:
    '''
    SettingModule: Abstract Class
    Entries: 
    '''
    
    setting_name = None
    setting_description = None
    use_cuda = False
    
    dataset = None
    method = None
    result = None
    evaluate = None

    def __init__(self, sName=None, sDescription=None):
        self.setting_name = sName
        self.setting_description = sDescription
    
    def prepare(self, sDataset, sMethod, sResult, sEvaluate, use_cuda):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        self.use_cuda = use_cuda

    def print_setup_summary(self):
        print('Dataset:', self.dataset.dataset_name, ', Method:', self.method.method_name,
              ', Setting:', self.setting_name, ', Result:', self.result.result_name, ', Evaluation:', self.evaluate.evaluate_name)

    @abc.abstractmethod
    def load_run_save_evaluate(self):
        return

