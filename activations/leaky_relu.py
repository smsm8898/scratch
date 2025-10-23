import numpy as np

name = "leaky_relu"
ymin = - np.inf
ymax = np.inf

def func(x, alpha=0.01):
    return np.maximum(alpha*x, x)

def derivative(x, alpha=0.01):
    return np.where(x > 0, 1., alpha)