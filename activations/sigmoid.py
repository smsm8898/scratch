import numpy as np

name = "sigmoid"
ymax = 1
ymin = 0

def func(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return func(x) * (1 - func(x))