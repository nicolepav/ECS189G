'''
MethodModule class for Multi-Layer Perceptron model on Classification Task
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch, sklearn
from torch import nn
from .base.method import method
from .evaluation import Evaluate_Accuracy
from sklearn.metrics import classification_report

class MLP(method, nn.Module):
    '''
    TBD
    '''

    # Initialization function
    def __init__(self, input_size, layers_data, learning_rate=0.001, optimizer=torch.optim.Adam,
                 loss_function=nn.CrossEntropyLoss(), epoch=500, training_loss = []):
        super(MLP, self).__init__("Multi-Layer Perceptron", "Classification Task")
        nn.Module.__init__(self)

        # Construct layers
        self.input_size = input_size
        self.layers = nn.ModuleList()
        self.training_loss = []
        for neuron_size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, neuron_size))
            input_size = neuron_size
            if isinstance(activation, nn.Module):
                self.layers.append(activation)

        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        self.loss_function = loss_function
        self.max_epoch = epoch
        self.training_loss = []


    # Forward propagation function
    def forward(self, x):
        y_pred = x
        for layer in self.layers:
            y_pred = layer(y_pred)

        return y_pred

    # Train function
    def train(self, X, y):
        # Initialize optimizer and evaluator
        minibatch = 6000
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # An iterative gradient updating process with mini-batch
        for epoch in range(self.max_epoch):
            # mini-batch
            for i in range(0, X.shape[0], minibatch):
                X_mini = X[i:i + minibatch]
                y_mini = y[i:i + minibatch]
                # Forward step
                y_pred = self.forward(X_mini)

                # Calculate the training loss
                y_true = y_mini
                train_loss = self.loss_function(y_pred, y_true)
                self.optimizer.zero_grad()
                # Backward step: error backpropagation
                train_loss.backward()

                # Update the variables according to the optimizer and the gradients calculated by the above loss function
                self.optimizer.step()

                #self.accuracy.append(accuracy_evaluator.evaluate())
            if epoch % 100 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch + 1, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())
            self.training_loss.append(train_loss.item())

        """for epoch in range(self.max_epoch):
            # Forward step
            y_pred = self.forward(X)

            # Calculate the training loss
            y_true = y
            train_loss = self.loss_function(y_pred, y_true)
            self.optimizer.zero_grad()
            # Backward step: error backpropagation
            train_loss.backward()

            # Update the variables according to the optimizer and the gradients calculated by the above loss function
            self.optimizer.step()
            self.training_loss.append(train_loss.item())
            # self.accuracy.append(accuracy_evaluator.evaluate())
            if epoch % 100 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch + 1, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())"""

        # ---- Performance Metrics------------------------------
        print(classification_report(y_true, y_pred.max(1)[1]))

    # Test function
    def test(self, X):
        # Forward step
        y_pred = self.forward(X)

        # Handling output layer activation: softmax:
        # Convert the probability distributions to the corresponding labels
        # Instances will get the labels corresponding to the largest probability

        # ---- Performance Metrics------------------------------
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

