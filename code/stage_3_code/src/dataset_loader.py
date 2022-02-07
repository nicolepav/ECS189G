'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.dataset import dataset
import numpy as np
import pickle


class Dataset_Loader(dataset):

    usage = None
    
    def __init__(self, dName=None, dDescription=None, dPath=None):
        super().__init__(dName, dDescription, dPath)

    def __len__(self):
        return len(self.data['X'])

    def load(self, train=True):
        self.usage = 'train' if train else 'test'
        print('Loading ' + self.usage + 'ing data...')
        X = []
        y = []
        try:
            f = open(self.dataset_source_file_path, 'rb')
            data = pickle.load(f)
            f.close()
        except FileNotFoundError:
            print("Dataset file not found")
            exit(1)

        for pair in data[self.usage]:
            X.append(pair['image'])
            y.append(pair['label'])

        self.data = {'X': np.array(X), 'y': np.array(y)}
        print('Dataset from', self.dataset_source_file_path, 'of size', len(self.data['X']), 'is loaded successfully!')

