import numpy as np

class BinaryCrossEntropy:
    name = "bce"
    def __init__(self, eps=1e-7):
        self.eps = eps
        
    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        left_hand = y_true * np.log(y_pred)
        right_hand = (1 - y_true) * np.log(1 - y_pred)
        loss = -(left_hand + right_hand)
        return np.mean(loss)

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        left_hand =  - y_true / y_pred
        right_hand =  (1 - y_true) / (1 - y_pred)
        grad = left_hand + right_hand
        return grad / len(y_true)

class CategoricalCrossEntropy:
    name = "cce"
    def __init__(self, eps=1e-7):
        self.eps = eps

    def forward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        loss = - np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)

    def backward(self, y_true, y_pred):
        y_pred = np.clip(y_pred, self.eps, 1 - self.eps)
        grad = - y_true / y_pred
        return grad / len(y_true)