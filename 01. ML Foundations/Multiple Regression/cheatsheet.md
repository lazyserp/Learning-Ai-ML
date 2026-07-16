# Multiple Regression Cheatsheet (2-Minute Revision)

---

## 1. Core Workflow

```
Problem: Predictions depend on multiple features
   │
   ▼
Approach: Vectorized Hyperplane Equation (Y_pred = X . W + b)
   │
   ▼
Error Distribution: Transpose feature matrix (X^T . Error)
   │
   ▼
Regularization/Pruning: Check for Multicollinearity (VIF)
```

---

## 2. Key Mathematics
* **Vectorized Hypothesis**: 
  $$\hat{Y} = X \cdot W + b$$
  - $X$: Shape $(n, d)$
  - $W$: Shape $(d, 1)$
  - $b$: Scalar bias (or added as a column of 1s in $X$)
* **Weight Gradients ($dW$)**: 
  $$dW = \frac{2}{n} X^T \cdot (\hat{Y} - Y)$$
* **Bias Gradients ($db$)**: 
  $$db = \frac{2}{n} \sum (\hat{y}_i - y_i)$$

---

## 3. High-Frequency Facts
* **Multicollinearity**: Correlated features lead to unstable, uninterpretable weights. Check with VIF.
* **Vectorization speed**: Matrix operations leverage optimized BLAS engines. Avoid writing Python loops for sample evaluation.
* **Feature Engineering**: Combining columns (e.g. ratios, multipliers) helps simple models learn complex interactions.
