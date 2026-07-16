# Logistic Regression Interview Questions

This document prepares you for high-frequency questions regarding Logistic Regression in technical interviews.

---

## 1. Questions & Answers

### Q. Why can't we use Linear Regression for Classification problems?
* **Answer**:
  1. **Unbounded outputs**: Linear regression outputs values from $-\infty$ to $+\infty$, which do not represent probabilities.
  2. **Sensitivity to outliers**: Outliers far from the decision boundary change the slope of the line, shifting the classification threshold and rendering correct predictions incorrect.
  3. **Non-convexity & Loss**: Classification targets are discrete $\{0,1\}$. Fitting a line to discrete targets using MSE loss yields a non-convex optimization surface when combined with non-linear mapping, leading to poor parameter convergence.

### Q. Why is Cross-Entropy loss preferred over Mean Squared Error (MSE) for Logistic Regression?
* **Answer**:
  1. **Convexity**: BCE loss is convex with respect to weights when combined with the Sigmoid activation. This guarantees that gradient descent will find the global minimum. MSE combined with Sigmoid is non-convex and has many local minima.
  2. **Gradient Vanishing**: The derivative of MSE with respect to weights contains a term $\sigma'(z) = \sigma(z)(1-\sigma(z))$. If the prediction is highly wrong (e.g. $p=0.99$ when $y=0$), $\sigma'(z) \to 0$. This causes the gradient to vanish, meaning the model stops learning. BCE loss cancels out this term, ensuring steep gradients and fast recovery from bad mistakes.

### Q. What is the relation between odds and probability in Logistic Regression?
* **Answer**:
  - **Odds** represents the ratio of the probability of success to the probability of failure:
    $$\text{Odds} = \frac{p}{1-p}$$
  - The **Logit** function is the natural logarithm of the odds (log-odds):
    $$\text{Logit}(p) = \log\left(\frac{p}{1-p}\right)$$
  - Logistic Regression models the log-odds as a linear combination of inputs:
    $$\log\left(\frac{p}{1-p}\right) = W \cdot X + b$$

---

## 2. Common Beginner Mistakes
1. **Using MSE Loss**: Trying to train a custom neural net or classification optimizer using distance-based MSE loss.
2. **Forgetting to check probability output**: Reading class 0/1 predictions without checking model confidence (probabilities).
3. **Evaluating heavily imbalanced datasets with Accuracy**: Accuracy is misleading on highly skewed data (e.g., fraud occurring in 0.1% of transactions). Use Precision, Recall, and ROC-AUC instead.
