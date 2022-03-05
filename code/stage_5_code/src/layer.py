'''
Layer module class for Graph Convolutional Networks on semi-supervised classification

@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))

        # Parameter Initialization
        std_v = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-std_v, std_v)
        self.bias.data.uniform_(-std_v, std_v)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)

        return output + self.bias

    # Self-introduction
    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' in-features -> ' \
               + str(self.out_features) + ' out-features)'

