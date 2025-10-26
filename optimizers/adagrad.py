import numpy as np

class Adagrad:
    name = "adagrad"
    def __init__(self, lr, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = 0 # inital gradient

    def update(self, w, grad):
        self.G = self.G + (grad ** 2)
        w = w - self.lr / (np.sqrt(self.G) + self.eps) * grad
        return w