import numpy as np

class RMSProp:
    name = "rmsprop"
    def __init__(self, lr, beta=0.9, eps=1e-8):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = 0

    def update(self, w, grad):
        self.s = self.beta * self.s + (1 - self.beta) * (grad**2)
        w = w - self.lr / (np.sqrt(self.s) + self.eps) * grad
        return w