# Polynomial Regression Cheatsheet (2-Minute Revision)

---

## 1. Core Workflow

```
Problem: Data trends are curved (non-linear)
   │
   ▼
Approach: Feature Engineering (Calculate x^2, x^3...)
   │
   ▼
Representation: Linear in Parameters (y = w1 * x + w2 * x^2 + b)
   │
   ▼
Evaluation: Monitor Train vs. Test Loss (Check for Overfitting)
```

---

## 2. Key Mathematics
* **Model Equation (degree $d$)**: 
  $$\hat{y} = w_1 x + w_2 x^2 + \dots + w_d x^d + b$$
* **Design Matrix ($X_{\text{poly}}$)**: 
  Each sample row contains powers $[x_i, x_i^2, \dots, x_i^d]$.
* **Linearity Rule**: 
  Linear Regression refers to linearity of coefficients ($W$), not inputs ($X$).

---

## 3. High-Frequency Facts
* **Underfitting**: Caused by too low polynomial degree. High bias.
* **Overfitting**: Caused by too high polynomial degree. High variance. Model oscillates wildly between data points.
* **Inference rule**: Test queries must pass through the exact same polynomial transformation pipeline before predicting.
