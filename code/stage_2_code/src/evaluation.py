'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from .base.evaluate import evaluate
from sklearn.metrics import accuracy_score


class Evaluate_Accuracy(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        return accuracy_score(self.data['true_y'].cpu(), self.data['pred_y'].cpu())

