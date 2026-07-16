# Polynomial Regression: Capturing Curves

This module covers Polynomial Regression, which allows linear models to represent non-linear relationships.

---

## 1. Problem
Real-world data trends are rarely straight lines. For example, the trajectory of a rocket, the growth of bacterial colonies, or the way house prices accelerate in high-demand zones are curved. 

## 2. Why Previous Algorithm Failed
Multiple Linear Regression ($y = w_1 x_1 + w_2 x_2 + b$) is constrained to fitting flat hyperplanes. If we force a flat plane onto curved data, the model fails to capture the curvature, resulting in **Underfitting** (the model is too simple to represent the pattern).

## 3. Intuition
Imagine you have a single feature $x$ (e.g., size). To represent a curve, we perform feature engineering: we generate higher-power features (e.g., $x^2$, $x^3$) and treat them as new inputs. 

Although the graph of the function is a curve, the mathematical model remains a linear combination of weights, allowing us to solve non-linear curves using standard linear solvers.

```
 Target (y)
   ^        *
   |       *
   |      *  <-- Polynomial Curve (y = w1*x + w2*(x^2) + b)
   |     *
   |    *
   +-----------------------> Input (x)
```
