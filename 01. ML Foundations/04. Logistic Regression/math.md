# Logistic Regression Mathematics

This document details the equations for Logistic Regression.

---

## 1. Sigmoid Function

Squashes raw score $z$ into probability $p$:
$$p = \sigma(z) = \frac{1}{1 + e^{-z}}$$

---

## 2. Binary Cross-Entropy Loss (Log Loss)

Measures classification error across $n$ samples:
$$L = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(p_i) + (1 - y_i) \log(1 - p_i) \right]$$

* **$y_i$**: Actual binary label ($0$ or $1$) for sample $i$.
* **$p_i$**: Predicted probability ($p_i \in [0, 1]$) for sample $i$.

---

## 3. Gradients and Updates

Despite the different loss function (Cross-Entropy instead of MSE) and hypothesis function (Sigmoid), when calculating the partial derivatives, the Sigmoid derivative combines elegantly with the log terms, yielding a gradient update formula identical in form to linear regression:

$$\frac{\partial L}{\partial W} = \frac{1}{n} X^T \cdot (P - Y)$$

$$\frac{\partial L}{\partial b} = \frac{1}{n} \sum_{i=1}^n (p_i - y_i)$$

* **$P$**: Vector of predicted probabilities of shape $(n, 1)$.
* **$Y$**: Vector of ground truth labels of shape $(n, 1)$.
* **$X$**: Feature matrix of shape $(n, d)$.

Weights are updated using Gradient Descent:
$$W \leftarrow W - \alpha \cdot \frac{\partial L}{\partial W}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$
