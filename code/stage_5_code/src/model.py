'''
MethodModule class for Convolutional Neural Network on Classification Task

@article{kipf2016semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  journal={arXiv preprint arXiv:1609.02907},
  year={2016}
}
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from .base.method import method
from .evaluation import evaluate_accuracy
from .layer import GraphConvolution

import torch.nn as nn
import torch.nn.functional as fun
from sklearn.metrics import classification_report


class GCN(method, nn.Module):

    # Initialization function
    def __init__(self, n_feat, n_hid, n_class, dropout, epoch, loss_function, optimizer, learning_rate, weight_decay):
        super(GCN, self).__init__('Graph Convolutional Networks', 'Semi-supervised Classification')
        nn.Module.__init__(self)

        self.gc1 = GraphConvolution(n_feat, n_hid)
        self.gc2 = GraphConvolution(n_hid, n_class)
        self.dropout = dropout

        self.max_epoch = epoch
        self.loss_function = loss_function
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.training_loss = []

    # Forward propagation function
    # relu –> dropout –> gc2 –> log softmax
    def forward(self, x, adj):
        x = fun.relu(self.gc1(x, adj))
        x = fun.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return fun.log_softmax(x, dim=1)


# Train function
def train(model, data_obj, fast_mode):

    # Initialize data components
    y_pred = []
    features = data_obj.data['graph']['X']
    labels = data_obj.data['graph']['y']
    adj = data_obj.data['graph']['utility']['A']
    idx_train = data_obj.data['train_test_val']['idx_train']
    idx_val = data_obj.data['train_test_val']['idx_val']

    for epoch in range(model.max_epoch):
        # Switch to training mode
        model.train()

        # Forward step
        y_pred = model(features, adj)

        # Calculate the training loss
        train_loss = model.loss_function(y_pred[idx_train], labels[idx_train])
        model.optimizer.zero_grad()

        # Backward step: error backpropagation
        train_loss.backward()

        # Update the variables according to the optimizer and the gradients calculated by the above loss function
        model.optimizer.step()

        print('Epoch:', epoch + 1, 'Train Accuracy:', evaluate_accuracy(y_pred[idx_train], labels[idx_train]).item(),
              'Train Loss:', train_loss.item())
        if not fast_mode:
            # Evaluate validation set performance separately, deactivates dropout during validation run.
            model.eval()
            valid_pred = model(features, adj)
            print('Validation Accuracy:', evaluate_accuracy(valid_pred[idx_val], labels[idx_val]).item())
        model.training_loss.append(train_loss.item())

    # ---- Performance Metrics------------------------------
    train_labels = labels[idx_train].cpu().detach().numpy()
    train_preds = y_pred[idx_train].max(1)[1].type_as(y_pred[idx_train]).cpu().detach().numpy()
    print(classification_report(train_labels, train_preds))


# Test function
def test(model, data_obj):
    model.eval()

    features = data_obj.data['graph']['X']
    labels = data_obj.data['graph']['y']
    adj = data_obj.data['graph']['utility']['A']
    idx_test = data_obj.data['train_test_val']['idx_test']

    y_pred = model(features, adj)

    test_labels = labels[idx_test].cpu().detach().numpy()
    test_labels = y_pred[idx_test].max(1)[1].type_as(y_pred[idx_test]).cpu().detach().numpy()
    print(classification_report(test_labels, test_labels))

    return evaluate_accuracy(y_pred[idx_test], labels[idx_test]).item()

