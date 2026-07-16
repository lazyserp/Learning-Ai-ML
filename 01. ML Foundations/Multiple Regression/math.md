# Multiple Regression Mathematics

This document details the matrix algebra used in Multiple Linear Regression.

---

## 1. Vectorized Hypothesis Function

For $d$ input features, predictions for a single sample are written as:
$$\hat{y} = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b$$

For $n$ samples, we represent features as a matrix $X$ of shape $(n, d)$, weights as a vector $W$ of shape $(d, 1)$, and bias $b$ as a scalar. The vectorized equation is:
$$\hat{Y} = X \cdot W + b$$

* **$\hat{Y}$**: Output vector of shape $(n, 1)$.
* **$X$**: Feature matrix of shape $(n, d)$.
* **$W$**: Weight vector of shape $(d, 1)$.
* **$b$**: Bias scalar.

---

## 2. Gradient Descent in Vector Form

The MSE loss remains the same:
$$L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

To calculate the gradient vector of the loss with respect to the weight vector $W$, we use:
$$\nabla_W L = \frac{2}{n} X^T \cdot (XW + b - Y)$$

* **$X^T$**: Transposed feature matrix of shape $(d, n)$.
* **$(XW + b - Y)$**: Prediction error vector of shape $(n, 1)$.
* **$\nabla_W L$**: Gradient vector of shape $(d, 1)$.

The updates are performed as:
$$W \leftarrow W - \alpha \cdot \nabla_W L$$
$$b \leftarrow b - \alpha \cdot \frac{2}{n} \sum_{i=1}^n (\hat{y}_i - y_i)$$
