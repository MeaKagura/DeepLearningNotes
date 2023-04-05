import torch
import numpy
from d2l import torch as d2l

# X = torch.normal(0, 1, (16, 2))
# w1 = torch.normal(0, 1, (2, 1))
# w2 = w1.reshape(2)
# b = torch.ones(1)
#
# y_hat1 = torch.matmul(X, w1) + b
# y_hat2 = torch.matmul(X, w2) + b
# print(y_hat1)
#
# print(w1)
# print(w1.sum())
# float(w1.sum())


def softmax(X):
    # X: [batch_size, num_class]
    X = torch.exp(X)
    partition = X.sum(1, keepdim=True)
    return X / partition


class Model:
    def __init__(self, num_inputs, num_outputs):
        self.w = torch.normal(0, 1, (num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def __call__(self, X):
        return softmax(torch.matmul(X.reshape((-1, self.w.shape[0])), self.w) + self.b)

    def get_params(self):
        return [self.w, self.b]


num_inputs = 784
num_outputs = 10
lr = 0.1
batch_size = 256
num_epochs = 20
model = Model(num_inputs, num_outputs)
print(len(model.get_params()))

num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
print(len([W, b]))