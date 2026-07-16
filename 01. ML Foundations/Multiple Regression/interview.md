# Multiple Regression Interview Questions

This document prepares you for high-frequency questions regarding Multiple Linear Regression in technical interviews.

---

## 1. Questions & Answers

### Q. What is Multicollinearity, and why is it dangerous?
* **Answer**: Multicollinearity occurs when two or more input features in a multiple regression model are highly linearly correlated. It is dangerous because:
  1. It makes the model weights ($W$) highly sensitive to minor changes in the training data, leading to unstable parameters.
  2. It obscures interpretability. The model cannot identify which of the correlated features is actually driving the target value.
  3. Mathematically, it makes the matrix $X^T X$ nearly singular (non-invertible), causing numerical instability in analytical solvers.

### Q. How do you identify and handle Multicollinearity in a dataset?
* **Answer**:
  - **Identify**:
    1. Look at correlation heatmaps to spot high pairwise correlation coefficients ($r > 0.8$).
    2. Calculate the **Variance Inflation Factor (VIF)**. A VIF value $> 5$ or $10$ indicates significant multicollinearity.
  - **Handle**:
    1. Remove one of the redundant, highly correlated features.
    2. Combine the features (e.g., compute their average or ratio).
    3. Use regularization (Ridge/Lasso) or dimensionality reduction (PCA).

### Q. Why is matrix transposition ($X^T$) used in vectorized gradient computation?
* **Answer**: 
  The prediction error vector $E = (\hat{Y} - Y)$ has shape $(n, 1)$. The feature matrix $X$ has shape $(n, d)$. To compute the gradients for our weights $W$ (which has shape $(d, 1)$), we must distribute the errors across each feature. Mathematically, this requires multiplying each feature column by the errors. By transposing $X$ to shape $(d, n)$, the matrix dot product $X^T \cdot E$ yields a vector of shape $(d, 1)$, matching the parameters vector shape precisely.

---

## 2. Common Beginner Mistakes
1. **Forgetting to check for collinearity**: Putting highly redundant features (like `area_in_inches` and `area_in_feet`) into the model.
2. **Dimension mismatch in Matrix multiplication**: Multiplying `np.dot(W, X)` instead of `np.dot(X, W)`.
3. **Failing to scale features of vastly different units**: E.g., area ($2500$ sq ft) and bedrooms ($3$), causing gradients to be dominated by the larger feature.
