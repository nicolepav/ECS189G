'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.result import result
import pickle


class Result_Loader(result):
    data = None
    fold_count = None
    result_destination_file_path = None
    
    def load(self):
        print('loading results...')
        f = open(self.result_destination_file_path, 'rb')
        self.data = pickle.load(f)
        f.close()

