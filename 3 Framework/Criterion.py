import numpy as np


class Criterion:
    def __init__(self):
        pass

    def loss(self, input, real_values):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class MSELoss(Criterion):
    def __init__(self):
        super().__init__()

    def loss(self, input, real_values):
        self.input = input
        self.real_values = real_values
        loss = np.mean((self.real_values - self.input) * (self.real_values - self.input))
        return loss

    def backward(self):
        return 2 * self.input - 2 * self.real_values

def softmax(xs):
    xs = np.subtract(xs, xs.max(axis=1, keepdims=True))
    xs = np.exp(xs) / np.sum(np.exp(xs), axis=1, keepdims=True)
    return xs

class CrossEntropyLoss(Criterion):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def loss(self, input, real_values):
        self.input = input
        self.real_values = real_values
        loss = -np.sum(self.real_values * np.log(np.clip(self.input, self.eps, 1/self.eps)))
        return loss / len(input)

    def backward(self):
        #return -softmax(self.input) + self.real_values
        return -self.real_values / (self.input + self.eps)

