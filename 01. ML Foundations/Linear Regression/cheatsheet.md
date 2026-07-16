# Linear Regression Cheatsheet (2-Minute Revision)

---

## 1. Core Workflow

```
Problem: Continuous Prediction
   │
   ▼
Approach: Fit a Straight Line (y = w * x + b)
   │
   ▼
Error Measurement: Mean Squared Error (MSE)
   │
   ▼
Adjustment: Gradient Descent Updates (w <- w - alpha * dw)
```

---

## 2. Key Mathematics
* **Hypothesis**: 
  $$\hat{y} = w \cdot x + b$$
* **Loss Function (MSE)**: 
  $$L = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$
* **Weight Update**: 
  $$w \leftarrow w - \alpha \cdot \left( -\frac{2}{n} \sum_{i=1}^n x_i (y_i - \hat{y}_i) \right)$$
* **Bias Update**: 
  $$b \leftarrow b - \alpha \cdot \left( -\frac{2}{n} \sum_{i=1}^n (y_i - \hat{y}_i) \right)$$

---

## 3. High-Frequency Facts
* **Divergence**: Caused by a high learning rate ($\alpha$) or poorly scaled features.
* **Ordinary Least Squares (OLS)**: The analytical closed-form method used by scikit-learn. It avoids iteration but scales poorly for high-dimensional matrices ($O(D^3)$).
* **Baseline shift**: Solved by adding the bias term ($b$), which shifts the prediction line off the origin.
