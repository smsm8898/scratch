import numpy as np

name = "gelu"
ymin = -np.inf
ymax = np.inf

def func(x):
    # 근사식
    return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

def derivative(x):
    # 근사식에서 도함수를 구함
    t = np.sqrt(2/np.pi) * (x + 0.044715 * x**3)
    tanh_t = np.tanh(t)
    sech2_t = 1 - tanh_t **2 # sech^2(t) = 1 - tanh^2(t)
    return 0.5 * (1 + tanh_t + 0.5 * x * sech2_t * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2) )