import numpy as np

name = "relu"
ymin = 0
# ymax = 무한대

def func(x):
    return np.maximum(0, x)

def derivative(x):
    return np.float64(x > 0)