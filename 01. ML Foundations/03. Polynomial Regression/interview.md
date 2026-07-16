# Polynomial Regression Interview Questions

This document prepares you for high-frequency questions regarding Polynomial Regression in technical interviews.

---

## 1. Questions & Answers

### Q. Is Polynomial Regression a linear or non-linear model?
* **Answer**: It is mathematically a **Linear** model. The definition of a linear model is that it is linear in the **parameters (weights $w_i$)**, not in the features ($x$). Because the equation is a linear combination of coefficients ($y = w_1 x + w_2 x^2 + b$), standard linear optimization algorithms (such as OLS and Gradient Descent) can solve it.

### Q. What is the difference between Underfitting and Overfitting, and how do they relate to polynomial degrees?
* **Answer**:
  - **Underfitting (High Bias)**: The model is too simple to capture the underlying pattern. This happens when the polynomial degree is too low (e.g., using degree=1 on quadratic data). The model will have high error on both training and test sets.
  - **Overfitting (High Variance)**: The model is too complex and fits training noise instead of the pattern. This occurs when the polynomial degree is too high relative to the amount of data (e.g., degree=10 on 6 samples). The model will have zero error on the training set but huge error on the test set.

### Q. How do you combat overfitting in high-degree Polynomial Regression?
* **Answer**:
  1. **Regularization**: Use Ridge (L2) or Lasso (L1) regression to penalize large weights.
  2. **Reduce Polynomial Degree**: Select a lower degree based on cross-validation performance.
  3. **Collect More Data**: Adding more training samples helps the model generalize the overall curve instead of fitting local noise.

---

## 2. Common Beginner Mistakes
1. **Using extremely high degrees**: E.g., choosing degree=12 on small datasets, causing massive runaway predictions outside the training range.
2. **Forgetting to transform test/query inputs**: Sending raw inputs directly to the model during inference.
3. **Double fitting**: Calling `.fit_transform()` on the test dataset, which shifts the normalization parameters. Always use `.transform()`.
