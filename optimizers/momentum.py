class Momentum:
    name = "momentum"
    def __init__(self, lr, beta=0.9):
        self.lr = lr
        self.beta = beta
        self.v = 0 # initial velocity

    def update(self, w, grad):
        self.v = self.beta * self.v + (1 - self.beta) * grad
        w = w - self.lr * self.v
        return w