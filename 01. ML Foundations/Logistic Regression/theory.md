# Logistic Regression Theory

This document details the concepts underpinning Logistic Regression.

---

## 1. Core Concepts

### Logit ($z$)
* **Problem**: Raw feature combinations $w \cdot x + b$ can be any value, but probability must be between $0$ and $1$.
* **Solution**: The linear score $z$ is called the **Logit** (or log-odds). It represents the unnormalized logarithm of the odds of the class being $1$.

### Sigmoid Function ($\sigma$)
* **Problem**: Map any real number $z \in (-\infty, \infty)$ to a valid probability range $[0, 1]$.
* **Solution**: The Sigmoid function:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
  * Large positive $z \to 1$
  * Large negative $z \to 0$
  * $z=0 \to 0.5$

### Decision Threshold
* **Problem**: Convert continuous probabilities ($[0, 1]$) into discrete classes ($0$ or $1$).
* **Solution**: Define a threshold (default $0.5$):
  $$\text{Prediction} = \begin{cases} 
  1 & \text{if } p \ge 0.5 \\
  0 & \text{if } p < 0.5 
  \end{cases}$$

### Binary Cross-Entropy (Log Loss)
* **Problem**: Mean Squared Error (MSE) is non-convex when combined with the Sigmoid function, causing gradient descent to get stuck in local minima. Additionally, MSE does not penalize confident wrong predictions heavily.
* **Solution**: Use Binary Cross-Entropy. It measures classification error by computing the negative log probability of the true label:
  $$\text{Loss} = - [y \log(p) + (1-y) \log(1-p)]$$
  * If true label $y=1$: Loss is $-\log(p)$. If predicted $p \to 0$ (highly confident mistake), loss grows to $\infty$.
  * If true label $y=0$: Loss is $-\log(1-p)$. If predicted $p \to 1$ (highly confident mistake), loss grows to $\infty$.
