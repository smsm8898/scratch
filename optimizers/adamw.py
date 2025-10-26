import numpy as np

class AdamW:
    name = "adamw"
    def __init__(self, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = 0
        self.v = 0
        self.t = 0
        

    def update(self, w, grad):
        self.t += 1

        # moment
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * (grad**2)

        # bias correction
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # weight update
        w = w - self.lr * (m_hat / (np.sqrt(v_hat) + self.eps) + self.weight_decay * w)
        return w