import numpy as np

class MSE:
    name = "mse"
    def forward(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def backward(self, y_true, y_pred):
        m = len(y_true)
        return -2 * (y_true - y_pred) / m
    
class MAE:
    name = "mae"
    def forward(self, y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def backward(self, y_true, y_pred):
        m = len(y_true)
        return -np.sign(y_true, y_pred) / m
    
class Huber:
    name = "huber"
    def __init__(self, delta=0.1):
        self.delta = delta

    def forward(self, y_true, y_pred):
        diff = y_true - y_pred
        condition = np.abs(diff) <= self.delta
        mse = 0.5 * diff **2 # small
        mae = self.delta * np.abs(diff) - 0.5 * self.delta # large
        loss = np.where(condition, mse, mae)
        return np.mean(loss)
    
    def backward(self, y_true, y_pred):
        diff = y_true - y_pred
        condition = np.abs(diff) <= self.delta
        mse = - diff # small
        mae = - self.delta * np.sign(diff) # large
        grad = np.where(condition, mse, mae)
        return grad / len(y_true)
