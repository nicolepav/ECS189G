'''
MethodModule class for Multi-Layer Perceptron model on Classification Task
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

import torch
from torch import nn
from .base.method import method
from .evaluation import Evaluate_Accuracy


class MLP(method, nn.Module):
    '''
    TBD
    '''

    # Initialization function
    def __init__(self, input_size, layers_data, learning_rate=0.01, optimizer=torch.optim.Adam,
                 loss_function=nn.CrossEntropyLoss(), epoch=500):
        super(MLP, self).__init__("Multi-Layer Perceptron", "Classification Task")
        nn.Module.__init__(self)

        # Construct layers
        self.input_size = input_size
        self.layers = nn.ModuleList()
        for neuron_size, activation in layers_data:
            self.layers.append(nn.Linear(input_size, neuron_size))
            input_size = neuron_size
            if isinstance(activation, nn.Module):
                self.layers.append(activation)

        self.learning_rate = learning_rate
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        self.loss_function = loss_function
        self.max_epoch = epoch

    # Forward propagation function
    def forward(self, x):
        y_pred = x
        for layer in self.layers:
            y_pred = layer(y_pred)

        return y_pred

    # Train function
    def train(self, X, y):
        # Initialize optimizer and evaluator
        self.optimizer.zero_grad()
        accuracy_evaluator = Evaluate_Accuracy('training evaluator', '')

        # An iterative gradient updating process without mini-batch
        for epoch in range(self.max_epoch):
            # Forward step
            y_pred = self.forward(X)

            # Calculate the training loss
            y_true = y
            train_loss = self.loss_function(y_pred, y_true)

            # Backward step: error backpropagation
            train_loss.backward()

            # Update the variables according to the optimizer and the gradients calculated by the above loss function
            self.optimizer.step()

            if epoch % 100 == 0 or epoch == self.max_epoch - 1:
                accuracy_evaluator.data = {'true_y': y_true, 'pred_y': y_pred.max(1)[1]}
                print('Epoch:', epoch + 1, 'Accuracy:', accuracy_evaluator.evaluate(), 'Loss:', train_loss.item())

    # Test function
    def test(self, X):
        # Forward step
        y_pred = self.forward(X)

        # Handling output layer activation: softmax:
        # Convert the probability distributions to the corresponding labels
        # Instances will get the labels corresponding to the largest probability
        return y_pred.max(1)[1]

    # Run function
    def run(self):
        print('Method Running...')
        print('--Start Training...')
        self.train(self.data['train']['X'], self.data['train']['y'])
        print('--Start Testing...')
        pred_y = self.test(self.data['test']['X'])

        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}

