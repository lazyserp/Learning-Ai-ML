# Multiple Regression Theory

This document outlines the core concepts of Multiple Linear Regression.

---

## 1. Core Concepts

### Vectorization
* **Problem**: Writing explicit loops to compute $y = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$ for millions of samples is slow in languages like Python.
* **Solution**: Stack features into a 2D matrix $X$ (samples $\times$ features) and weights into a vector $W$. Use high-performance vectorized operations (such as BLAS-optimized matrix multiplication) to compute predictions:
  $$\hat{Y} = X \cdot W + b$$

### Multicollinearity
* **Problem**: When input features are highly correlated with each other (e.g., `house_area_sq_ft` and `house_area_sq_meters`), the model cannot distinguish their individual effects. The weights become unstable and sensitive to minor changes in data, which hurts model interpretability.
* **Solution**: Identify correlated features using correlation matrices or Variance Inflation Factor (VIF), and remove redundant inputs (Feature Selection).

### Feature Engineering
* **Problem**: Raw features might have non-linear or interactive relationships (e.g. number of bedrooms relative to size is more important than size alone).
* **Solution**: Transform or combine raw inputs (e.g., `ratio = bedrooms / area`) to create new inputs that expose patterns more clearly.
