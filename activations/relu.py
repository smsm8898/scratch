import numpy as np

name = "relu"
ymin = 0
ymax = np.inf

def func(x):
    return np.maximum(0, x)

def derivative(x):
    return np.where(x > 0, 1., 0.)