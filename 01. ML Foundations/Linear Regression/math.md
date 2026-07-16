# Linear Regression Mathematics

This document defines the mathematical equations governing Simple Linear Regression.

---

## 1. Hypothesis Function

Our model predicts the target output $\hat{y}$ using a linear combination of the input $x$:
$$\hat{y} = w \cdot x + b$$

* **$\hat{y}$** (y-hat): Predicted continuous output.
* **$x$**: Input feature.
* **$w$**: Model weight (slope).
* **$b$**: Model bias (y-intercept).

---

## 2. Mean Squared Error (MSE) Loss

To evaluate the model's parameters, we measure the average squared difference between actual target values $y$ and predictions $\hat{y}$:
$$L(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

* **$L(w, b)$**: Total loss value.
* **$n$**: Number of training samples.
* **$y_i$**: Actual target label for sample $i$.
* **$\hat{y}_i$**: Predicted value for sample $i$.

---

## 3. Gradients & Updates

To minimize $L(w, b)$, we find the partial derivatives of the loss function with respect to $w$ and $b$ using the chain rule:

$$\frac{\partial L}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i \cdot (y_i - \hat{y}_i)$$

$$\frac{\partial L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$

We update the weights using Gradient Descent:
$$w \leftarrow w - \alpha \cdot \frac{\partial L}{\partial w}$$
$$b \leftarrow b - \alpha \cdot \frac{\partial L}{\partial b}$$

* **$\alpha$**: Learning rate.
