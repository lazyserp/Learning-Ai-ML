# Polynomial Regression Mathematics

This document details the mathematical expansion formulas for Polynomial Regression.

---

## 1. Mathematical Formulation

For a single input feature $x$, a polynomial model of degree $d$ is represented as:
$$\hat{y} = w_1 x + w_2 x^2 + w_3 x^3 + \dots + w_d x^d + b$$

* **$\hat{y}$**: Predicted value.
* **$x, x^2, \dots, x^d$**: Feature powers generated from the base input feature.
* **$w_1, w_2, \dots, w_d$**: Learnable parameters representing the coefficient of each polynomial degree.
* **$b$**: Bias term.

---

## 2. Matrix Vectorization

To solve this model, we construct the design matrix $X_{\text{poly}}$ of shape $(n, d)$:
$$X_{\text{poly}} = \begin{bmatrix} 
x_1 & x_1^2 & \dots & x_1^d \\
x_2 & x_2^2 & \dots & x_2^d \\
\vdots & \vdots & \ddots & \vdots \\
x_n & x_n^2 & \dots & x_n^d 
\end{bmatrix}$$

We can then solve this using standard linear regression matrix operations:
$$\hat{Y} = X_{\text{poly}} \cdot W + b$$
$$W \leftarrow (X_{\text{poly}}^T X_{\text{poly}})^{-1} X_{\text{poly}}^T Y$$
