import numpy as np

class Module:
    def __init__(self):
        self._trainable = True

    def forward(self, input):
        pass

    def backward(self, input, grad):
        pass

    def parameters(self):
        return [0, 0]

    def grad_parameters(self):
        return [0, 0]

    def train(self):
        self._trainable = True

    def eval(self):
        self._trainable = False



class Linear(Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.W = np.random.randn(input_dim, output_dim)
        self.b = np.random.randn(1, output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, input):
        self.output = (input @ self.W) + self.b
        return self.output

    def backward(self, input, grad):
        self.b_grad = np.mean(grad, axis = 0)
        self.W_grad = (input.T @ grad) / input.shape[0]
        return grad @ self.W.T

    def parameters(self):
        return [self.W, self.b]

    def grad_parameters(self):
        return [self.W_grad, self.b_grad]

class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        self.output = input
        for layer in self.layers:
            self.output = layer.forward(self.output)
        return self.output

    def backward(self, input, grad):
        for i in range(len(self.layers) - 1, 0, -1):
            grad = self.layers[i].backward(self.layers[i - 1].output, grad)
        grad = self.layers[0].backward(input, grad)
        return grad

    def parameters(self):
        answer = []
        for layer in self.layers:
            answer += layer.parameters()
        return answer

    def grad_parameters(self):
        answer = []
        for layer in self.layers:
            answer += layer.grad_parameters()
        return answer

    def train(self):
        for layer in self.layers:
            layer.train()

    def eval(self):
        for layer in self.layers:
            layer.eval()


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-9

    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def backward(self, input, grad):
        answer = np.array(grad, copy = True)
        answer[self.output <= self.eps] = 0
        return answer

class LeakyReLU(Module):
    def __init__(self, alpha):
        super(LeakyReLU, self).__init__()
        self.alpha = alpha
        self.eps = 1e-9

    def forward(self, input):
        self.output = np.maximum(input, self.alpha * input)
        return self.output

    def backward(self, input, grad):
        grad_coefs = (input > 0) + self.alpha * (input <= 0)
        return grad_coefs * grad

class SoftMax(Module):
    def __init__(self):
        super(SoftMax, self).__init__()

    def forward(self, input):
        self.eps = 1e-9
        self.input = input
        self.output = np.subtract(self.input, self.input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output) / np.sum(np.exp(self.output), axis=1, keepdims=True)
        return self.output

    def backward(self, input, grad):
        batch_size = grad.shape[0]
        num_elements = grad.shape[1]

        coef_matrix = np.array([-self.output.reshape(batch_size, num_elements, 1)[i] @ self.output.reshape(batch_size, 1, num_elements)[i] for i in range(batch_size)])
        coef_matrix = -self.output.reshape(batch_size, num_elements, 1) @ self.output.reshape(batch_size, 1, num_elements)

        eye_matrixes = np.zeros((batch_size, num_elements, num_elements))
        eye_matrixes[:, np.arange(num_elements), np.arange(num_elements)] = self.output

        coef_matrix += eye_matrixes
        answer = coef_matrix.reshape(batch_size, num_elements, num_elements) @ grad.reshape(batch_size, num_elements, 1)
        return answer.reshape(batch_size, num_elements)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        self.output = 1.0 / (1.0 + np.exp(-input))
        return self.output

    def backward(self, input, grad):
        return self.output * (1 - self.output) * grad

class BatchNorm(Module):
    def __init__(self):
        super(BatchNorm, self).__init__()
        self.eps = 1e-5

    def forward(self, input):
        self.mean = input.mean()
        self.var = input.var()
        self.output = (input - self.mean) / np.sqrt(self.var + self.eps)
        return self.output

    def backward(self, input, grad):
        return grad / np.sqrt(self.var + self.eps)


class Dropout(Module):
    def __init__(self, alpha = 0.1):
        super(Dropout, self).__init__()
        self.alpha = alpha

    def forward(self, input):
        if self._trainable:
            self.mask = np.random.random(input.shape) > self.alpha
            self.output = input * self.mask
        else:
            self.output = input
        return self.output

    def backward(self, input, grad):
        if self._trainable:
            return self.mask * grad
        else:
            return grad * (1 - self.alpha)


