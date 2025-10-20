import numpy as np

name = "tanh"
ymax = 1
ymin = -1

def func(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def derivative(x):
    return 1 - func(x)**2