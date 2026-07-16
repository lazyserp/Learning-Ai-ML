# Logistic Regression Cheatsheet (2-Minute Revision)

---

## 1. Core Workflow

```
Problem: Binary Classification (discrete 0/1)
   │
   ▼
Linear Score (Logit): z = X . W + b
   │
   ▼
Squashing Activation: p = Sigmoid(z) = 1 / (1 + e^-z)
   │
   ▼
Error Metric: Binary Cross-Entropy Loss
   │
   ▼
Prediction: Threshold (p >= 0.5 -> class 1; else 0)
```

---

## 2. Key Mathematics
* **Sigmoid Activation**: 
  $$p = \frac{1}{1 + e^{-z}}$$
* **Binary Cross-Entropy Loss**: 
  $$L = -\frac{1}{n} \sum [y \log(p) + (1-y)\log(1-p)]$$
* **Update Step (Gradient Descent)**:
  $$W \leftarrow W - \alpha \cdot \left(\frac{1}{n} X^T \cdot (P - Y)\right)$$
  $$b \leftarrow b - \alpha \cdot \left(\frac{1}{n} \sum (p_i - y_i)\right)$$

---

## 3. High-Frequency Facts
* **Non-convexity of MSE**: Forcing MSE loss on Sigmoid predictions causes local minima and vanishing gradient issues. BCE solves this.
* **Interpretation of coef_**: If $w_i > 0$, increasing $x_i$ increases the probability of class 1.
* **Decision Boundary**: The plane where $z = 0$ (probability = $0.5$).
