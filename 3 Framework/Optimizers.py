from Layers import Module
import numpy as np

class Optimizer:
    def __init__(self, model: Module, lr: float):
        self.model = model
        self.lr = lr

    def step(self):
        raise NotImplementedError("Step is not implemented")


class SGDOptimizer(Optimizer):
    def __init__(self, model: Module, lr: float):
        super().__init__(model, lr)

    def step(self):
        for i, (w, grad_w) in enumerate(zip(self.model.parameters(), self.model.grad_parameters())):
            w -= self.lr * grad_w

class Adam(Optimizer):
    def __init__(self, model: Module, lr: float, k1: float = 0.9, k2: float = 0.99):
        super().__init__(model, lr)
        self.k1, self.k2 = k1, k2
        self.prev_vel = []
        self.prev_ac = []
        self.eps = 1e-9

    def step(self):
        start = len(self.prev_vel) == 0
        cur_v = []
        cur_a = []
        for i, (weights, grad_weights) in enumerate(zip(self.model.parameters(), self.model.grad_parameters())):
            if start:
                vel = grad_weights
                ac = grad_weights ** 2
                cur_v.append(vel)
                cur_a.append(ac)
                weights -= self.lr * vel / (self.eps + np.sqrt(ac))
            else:
                vel = grad_weights * (1 - self.k1) + self.k1 * self.prev_vel[i]
                ac = (grad_weights ** 2) * (1 - self.k2) + self.k2 * self.prev_ac[i]

                vel = vel
                ac = ac
                cur_v.append(vel)
                cur_a.append(ac)
                weights -= self.lr * vel / (self.eps + np.sqrt(ac))
        self.prev_vel = cur_v
        self.prev_ac = cur_a

