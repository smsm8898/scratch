# Scratch
Understanding and Implementing the fundamental building blocks of Deep Learning from original papers

## Prerequisities
- python >= 3.9
- numpy >= 2.0.0

---
## 🔸 Activation Functions

| No | Name | Paper Title | Year | Link |
|----|------|--------------|------|------|
| 1 | **Sigmoid** | – | – | – |
| 2 | **Tanh** | – | – | – |
| 3 | **ReLU** | *Rectified Linear Units Improve Restricted Boltzmann Machines* | 2010 | [PDF](https://www.cs.toronto.edu/~hinton/absps/reluICML.pdf) |
| 4 | **Leaky ReLU** | *Rectifier Nonlinearities Improve Neural Network Acoustic Models* | 2013 | [PDF](https://arxiv.org/abs/1303.2662) |
| 5 | **ELU (Exponential Linear Unit)** | *Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)* | 2015 | [arXiv:1511.07289](https://arxiv.org/abs/1511.07289) |
| 6 | **GELU (Gaussian Error Linear Unit)** | *Gaussian Error Linear Units (GELUs)* | 2016 | [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) |

---
## 🔸 Optimizers

| No | Name | Paper Title | Year | Link |
|----|------|--------------|------|------|
| 1 | **SGD** | – | – | – |
| 2 | **SGD + Momentum** | “A Method for Stochastic Optimization” | 1964 | [PDF](https://web.stanford.edu/class/ee398a/papers/polyak1964.pdf) |
| 3 | **Nesterov Accelerated Gradient (NAG)** | “A Method for Unconstrained Convex Minimization Problem with Convergence Rate O(1/k²)” | 1983 | [PDF](https://www.math.ku.dk/~rolf/teaching/10nesterov.pdf) |
| 4 | **AdaGrad** | “Adaptive Subgradient Methods for Online Learning and Stochastic Optimization” | 2011 | [PDF](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf) |
| 5 | **RMSProp** | – (Geoff Hinton lecture notes) | 2012 | [Link](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf) |
| 6 | **Adam** | *Adam: A Method for Stochastic Optimization* | 2015 | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| 7 | **AdamW** | *Decoupled Weight Decay Regularization* | 2017 | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) |
| 8 | **Nadam** | *Incorporating Nesterov Momentum into Adam* | 2016 | [arXiv:1609.04747](https://arxiv.org/abs/1609.04747) |
| 9 | **AdaMax** | *Adam: A Method for Stochastic Optimization* | 2015 | [arXiv:1412.6980](https://arxiv.org/abs/1412.6980) |
| 10 | **AMSGrad** | *On the Convergence of Adam and Beyond* | 2018 | [arXiv:1904.09237](https://arxiv.org/abs/1904.09237) |
| 11 | **RAdam** | *On the Variance of the Adaptive Learning Rate and Beyond* | 2019 | [arXiv:1908.03265](https://arxiv.org/abs/1908.03265) |
| 12 | **Yogi** | *Adaptive Methods for Nonconvex Optimization* | 2018 | [arXiv:1810.06801](https://arxiv.org/abs/1810.06801) |
| 13 | **Lion (EvoGrad)** | *Symbolic Discovery of Optimization Algorithms* | 2023 | [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) |


## 🔸 Normalization

| No | Name | Paper Title | Year | Link |
|----|------|--------------|------|------|
| 1 | **Batch Normalization** | *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* | 2015 | [arXiv:1502.03167](https://arxiv.org/abs/1502.03167) |
| 2 | **Layer Normalization** | *Layer Normalization* | 2016 | [arXiv:1607.06450](https://arxiv.org/abs/1607.06450) |

---

## 🔸 Regularization

| No | Name | Paper Title | Year | Link |
|----|------|--------------|------|------|
| 1 | **L1/L2 Regularization** | “A Simple Weight Decay Can Improve Generalization” (Andrew Krogh & John Hertz)| 1992 |[PDF](https://proceedings.neurips.cc/paper/1991/file/8eefcfdf5990e441f0fb6f3fad709e21-Paper.pdf)|
| 2 | **Dropout** | *Dropout: A Simple Way to Prevent Neural Networks from Overfitting* | 2014 | [arXiv:1207.0580](https://arxiv.org/abs/1207.0580) |
| 3|  **Label Smoothing** | “Rethinking the Inception Architecture for Computer Vision” (Szegedy et al.)| 2016 | [arXiv:1512.00567](https://arxiv.org/abs/1512.00567)|

---
<!-- ## 🔸 Weight Initialization

| No | Name | Paper Title | Year | Link |
|----|------|--------------|------|------|
| 1 | **Xavier Initialization** | *Understanding the Difficulty of Training Deep Feedforward Neural Networks* | 2010 | [arXiv:1006.0254](https://arxiv.org/abs/1006.0254) |
| 2 | **He Initialization**     | “Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification” (He et al.) | 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852)|
| 3 | **LSUV Initialization**   | “All You Need is a Good Init” (Mishkin & Matas) | 2016 | [arXiv:1511.06422](https://arxiv.org/abs/1511.06422) |
 -->

---

<!-- ## 🔸 Architecture Components

| No | Name | Paper Title | Year | Link |
|----|------|--------------|------|------|
| 1 | **Convolution (Conv2D)** | *ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)* | 2012 | [PDF](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) |
| 2 | **Pooling (Max/Avg)** | *LeNet-5, Gradient-Based Learning Applied to Document Recognition* | 1998 | [Link](http://yann.lecun.com/exdb/lenet/) |
| 3 | **Residual Connection (ResNet)** | *Deep Residual Learning for Image Recognition* | 2015 | [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) |

--- -->