'''
Concrete ResultModule class for a specific experiment ResultModule output
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.result import result
import pickle
import torch


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_file_path = None

    # Save predication result
    def save_learned_result(self):
        print('saving results...')
        if self.fold_count:
            path = self.result_destination_file_path + '_' + str(self.fold_count)
        else:
            path = self.result_destination_file_path
        f = open(path, 'wb')
        pickle.dump(self.data, f)
        f.close()

    # Save learned model
    def save_learned_model(self):
        torch.save(self.data, self.result_destination_file_path + '.pth')

