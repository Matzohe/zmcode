import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class LogisticRegression:
    def __init__(self, config):
        self.word_dim = int(config.HOTEL['word_dim'])
        self.max_len = int(config.HOTEL['max_sent_len'])
        self.lr = float(config.HOTEL['lr'])
        self.epochs = int(config.HOTEL['epoch'])
        
        self.linear = nn.Linear(self.word_dim * self.max_len, 2)

        self.optimizer = optim.Adam(self.linear.parameters(), lr=self.lr)

        self.save_path = config.HOTEL['weight_save_path']
        self.loss_fun = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)

    def sigmoid(self, x):
        return self.softmax(x)

    def loss(self, out, label):
        return self.loss_fun(out, label.view(-1))

    def forward(self, X):
        output = self.linear(X)
        return self.sigmoid(output)

    def gradient_descent(self, out, y):
        loss = self.loss(out, y)
        loss.sum().backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.sum().detach().data


    def train(self, train_iter, test_iter):
        train_losses = []
        test_losses = []
        for epoch in range(self.epochs):
            train_loss = 0
            n_samples = 0
            for data, label in train_iter:
                n_samples += len(data)
                out = self.forward(data)
                loss = self.gradient_descent(out, label)
                train_loss += loss.sum()

            train_loss /= n_samples
            test_loss = self.test(test_iter)

            train_losses.append(train_loss)
            test_losses.append(test_loss)

            print("epoch{}/{} training loss:{}, test loss:{}".format(epoch, self.epochs, train_loss, test_loss))

        return train_losses, test_losses

    def test(self, test_iter):
        test_loss = 0
        n_samples = len(test_iter) * test_iter.batch_size
        with torch.no_grad():
            for data, label in test_iter:
                out = self.forward(data)
                loss = self.loss(out, label)
                test_loss += loss.sum()

        test_loss /= n_samples
        return test_loss

    def test_accuracy(self, test_iter):
        rights = 0
        n_samples = len(test_iter) * test_iter.batch_size
        with torch.no_grad():
            for data, label in test_iter:
                out = self.forward(data)
                pred = torch.argmax(out, dim=1)
                rights += (pred == label).sum().item()

        return (rights / n_samples) * 100
