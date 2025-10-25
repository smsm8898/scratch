class SGD:
    name = "sgd"
    def __init__(self, lr):
        self.lr = lr

    def update(self, w, grad):
        w = w - self.lr * grad
        return w