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