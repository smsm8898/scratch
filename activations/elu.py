import numpy as np

name = "elu"
ymin = - np.inf
ymax = np.inf

def func(x, alpha=0.01):
    return np.where(x>0, x, alpha*(np.exp(x)-1))

def derivative(x, alpha=0.01):
    return np.where(x > 0, 1., alpha * np.exp(x))