'''
MethodModule class for Convolutional Neural Network on Classification Task
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch
from torch import nn
from .base.method import method
from .evaluation import Evaluate_Accuracy
from sklearn.metrics import classification_report


class CNN(method, nn.Module):
    '''
    TBD
    '''

    # Initialization function
    def __init__(self, layers_data, learning_rate=0.001, epoch=10, batch=5, optimizer=torch.optim.Adam,
                 loss_function=nn.CrossEntropyLoss(), device='cpu'):
        super(CNN, self).__init__("Multi-Layer Perceptron", "Classification Task")
        nn.Module.__init__(self)

        # Construct layers
        self.layers = nn.ModuleList()

        for layer in layers_data:
            self.layers.append(layer)

        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        self.loss_function = loss_function
        self.batch_size = batch
        self.max_epoch = epoch
        self.device = device
        self.training_loss = []


    # Forward propagation function
    def forward(self, x):
        first = True
        y_pred = x
        for layer in self.layers:
            if first and isinstance(layer, nn.Linear):
                y_pred = torch.flatten(y_pred, 1)
                first = False
            y_pred = layer(y_pred)

        return y_pred

    # Train function
    def train(self, X, y):
        # Initialize optimizer and evaluator
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # An iterative gradient updating process with mini-batch
        for epoch in range(self.max_epoch):
            # Mini-batch
            for i in range(0, X.shape[0], self.batch_size):
                X_mini = X[i:i + self.batch_size].to(self.device)
                y_true = y[i:i + self.batch_size].to(self.device)

                # Forward step
                y_pred = self.forward(X_mini)

                # Calculate the training loss
                train_loss = self.loss_function(y_pred, y_true).to(self.device)
                self.optimizer.zero_grad()

                # Backward step: error backpropagation
                train_loss.backward()

                # Update the variables according to the optimizer and the gradients calculated by the above loss function
                self.optimizer.step()

            if epoch % 10 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch + 1, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
            self.training_loss.append(train_loss.item())

        # ---- Performance Metrics------------------------------
        print(classification_report(y_true.cpu(), y_pred.max(1)[1].cpu()))

    # Test function
    def test(self, X):
        # Forward step
        y_pred = self.forward(X.to(self.device))

        # Handling output layer activation: softmax:
        # Convert the probability distributions to the corresponding labels
        # Instances will get the labels corresponding to the largest probability

        # ---- Performance Metrics------------------------------
        y_pred = y_pred.cpu()
        print(classification_report(self.data['test']['y'], y_pred.max(1)[1]))
        return y_pred.max(1)[1]

    # Run function
    def run(self):
        print('Method Running...')
        print('--Start Training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--Start Testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

