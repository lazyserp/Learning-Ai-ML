# Polynomial Regression Theory

This document outlines the core concepts of Polynomial Linear Regression.

---

## 1. Core Concepts

### Underfitting vs. Overfitting
* **Underfitting**: When the degree of the polynomial is too low (e.g., using $d=1$ on a quadratic curve). The model is too simple and has high bias.
* **Overfitting**: When the degree of the polynomial is too high (e.g., using $d=15$ on $4$ data points). The model will pass through every single point perfectly, capturing noise instead of the signal. This leads to low bias but high variance, causing poor generalization on new data.

```
Underfitting (d=1)        Balanced (d=2)        Overfitting (d=10)
      \   *                    *                    *    *
   *   \                     *   *                * / \  /
  *  *  \ *                *   *   *             * /   \/
```

### Linear in Parameters vs. Features
* **Key Concept**: A model is defined as "Linear" if the parameters (weights $w_i$) are linear. 
* Mathematically, $y = w_1 x_1 + w_2 x_1^2 + b$ is still a linear model because the weights are not exponentiated, passed through trig functions, or multiplied together. We treat $x_1^2$ simply as a new feature $x_2$.
