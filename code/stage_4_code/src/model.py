'''
MethodModule class for Convolutional Neural Network on Classification Task
'''

# Copyright (c) 2022-Current Jialiang Wang <jilwang804@gmail.com>
# License: TBD

from .base.method import method
from .evaluation import Evaluate_Accuracy

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report


class LSTM(method, nn.Module):

    # Initialization function
    def __init__(self, vocab_size, n_layers, embedding_dim, hidden_dim, output_dim, epoch, learning_rate, loss_function,
                 optimizer, device):
        super(LSTM, self).__init__("Long Short-Term Memory", "Classification Task")
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            bidirectional=True, dropout=0.5, batch_first=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)
        self.act = nn.Sigmoid()

        self.max_epoch = epoch
        self.loss_function = loss_function
        self.optimizer = optimizer(params=self.parameters(), lr=learning_rate)
        self.device = device
        self.training_loss = []
        self.evaluator = Evaluate_Accuracy('LSTM binary evaluator', '')

    # Forward propagation function
    def forward(self, x, x_len):
        embedding = self.embedding(x)
        packed_embedding = pack_padded_sequence(embedding, x_len, batch_first=True)
        packed_output, (hidden, cell) = self.lstm(packed_embedding)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        dense_outputs = self.fc(hidden)
        y_pred = self.act(dense_outputs).squeeze()

        return y_pred


# Train function
def train(model, train_iter):
    # Initialize loss function
    model.to(model.device)
    model.loss_function.to(model.device)

    epoch_y_pred = torch.tensor([])
    epoch_y_true = torch.tensor([])
    train_loss = None

    # An iterative gradient updating process with batch
    model.train()
    for epoch in range(model.max_epoch):
        # Reset records after completing an epoch
        epoch_y_pred = torch.tensor([])
        epoch_y_true = torch.tensor([])

        for batch in train_iter:
            text, text_len = batch.text
            text = text.to(model.device)
            text_len = text_len.to('cpu')
            y_true = batch.label.to(model.device)

            # Forward step
            y_pred = model(text, text_len)

            # Calculate the training loss
            train_loss = model.loss_function(y_pred, y_true)
            model.optimizer.zero_grad()

            # Backward step: error backpropagation
            train_loss.backward()

            # Update the variables according to the optimizer and the gradients calculated by the above loss function
            model.optimizer.step()
            epoch_y_pred = torch.cat((epoch_y_pred, torch.round(y_pred).to('cpu')))
            epoch_y_true = torch.cat((epoch_y_true, y_true.to('cpu')))

        #if epoch % 10 == 0 or epoch == model.max_epoch - 1:
        print('Epoch:', epoch + 1, 'Accuracy:', model.evaluator.binary_accuracy(epoch_y_pred, epoch_y_true).item(),
              'Loss:', train_loss.item())
        model.training_loss.append(train_loss.item())

    # ---- Performance Metrics------------------------------
    print(classification_report(epoch_y_true.numpy(), epoch_y_pred.detach().numpy()))


# Test function
def test(model, test_iter):
    epoch_y_pred = torch.tensor([])
    epoch_y_true = torch.tensor([])

    model.eval()
    with torch.no_grad():
        for batch in test_iter:
            text, text_len = batch.text
            text = text.to(model.device)
            text_len = text_len.to('cpu')
            y_true = batch.label.to(model.device)
            y_pred = model(text, text_len)
            epoch_y_pred = torch.cat((epoch_y_pred, torch.round(y_pred).to('cpu')))
            epoch_y_true = torch.cat((epoch_y_true, y_true.to('cpu')))

    # ---- Performance Metrics------------------------------
    print(classification_report(epoch_y_true.numpy(), epoch_y_pred.detach().numpy()))

    return model.evaluator.binary_accuracy(epoch_y_pred, epoch_y_true).item()

