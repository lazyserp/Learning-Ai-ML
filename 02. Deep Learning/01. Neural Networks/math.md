# Neural Networks Mathematics

This document details the equations governing Multi-Layer Perceptrons.

---

## 1. Single Layer Forward Computation

For a layer with input vector $A^{[l-1]}$ (activation from the previous layer), weight matrix $W^{[l]}$, and bias vector $b^{[l]}$:

$$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
$$A^{[l]} = g(Z^{[l]})$$

* **$Z^{[l]}$**: Linear logit score for layer $l$.
* **$W^{[l]}$**: Weight matrix of shape $(\text{neurons in layer } l, \text{neurons in layer } l-1)$.
* **$b^{[l]}$**: Bias vector of shape $(\text{neurons in layer } l, 1)$.
* **$A^{[l]}$**: Activated output vector for layer $l$.
* **$g(\cdot)$**: Non-linear activation function.

---

## 2. Standard Activation Functions

### Rectified Linear Unit (ReLU)
Converts negative values to zero, introducing sparse activation:
$$g(z) = \max(0, z)$$

### Sigmoid
Squashes values into range $[0, 1]$ (commonly used in binary outputs):
$$g(z) = \frac{1}{1 + e^{-z}}$$
