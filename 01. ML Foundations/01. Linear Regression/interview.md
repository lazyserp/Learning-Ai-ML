# Linear Regression Interview Questions

This document prepares you for high-frequency questions regarding Simple Linear Regression in technical interviews.

---

## 1. Questions & Answers

### Q. Why choose Gradient Descent over the closed-form Normal Equation?
* **Answer**: The closed-form solution (Normal Equation) requires computing $(X^T X)^{-1}$. Inverting a matrix has a computational complexity of $O(D^3)$ where $D$ is the number of features. When $D$ is large (e.g., $10^5$), computing the inverse is extremely slow and memory-intensive. Gradient Descent, with complexity $O(E \cdot N \cdot D)$, is much more efficient and scales to large datasets.

### Q. What happens to Gradient Descent if you do not scale features?
* **Answer**: If features are on different scales (e.g. house area $\approx 1000$ vs. bedrooms $\approx 2$), the contours of the loss function become highly elongated (elliptical). The gradient will point almost perpendicular to the path toward the minimum, causing weight updates to bounce back and forth instead of moving directly downhill. Scaling makes the landscape spherical, enabling faster and more direct convergence.

### Q. Why is Mean Squared Error (MSE) preferred over Mean Absolute Error (MAE) for regression?
* **Answer**: 
  1. **Differentiability**: MSE is smooth and differentiable everywhere. MAE has a sharp "V" shape and its derivative is undefined at $0$, which makes optimization harder.
  2. **Outlier Penalty**: MSE squares errors, meaning large errors are penalized much more heavily than small ones. This forces the model to prioritize reducing major mistakes.

---

## 2. Common Beginner Mistakes
1. **Forgetting Feature Scaling**: Forgetting to apply feature scaling, leading to gradient explosion.
2. **Predicting with Unscaled Inputs**: Training a model on scaled data (e.g., area scaled to $[0.5, 2.0]$) but querying it with raw data (e.g., area = $1800$), causing massive estimation errors.
3. **Choosing an Extreme Learning Rate**: Selecting a learning rate that is either too small (the model doesn't learn) or too large (the loss increases and diverges).
