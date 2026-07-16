# Logistic Regression: Binary Classification

This module covers Logistic Regression, which is the foundational algorithm for predicting discrete classes.

---

## 1. Problem
We need to predict a discrete binary category (e.g., whether an email is Spam [1] or Not Spam [0], or whether a transaction is Fraudulent [1] or Genuine [0]).

## 2. Why Previous Algorithm Failed
Linear/Polynomial Regression models predict continuous, unbounded values ($-\infty$ to $+\infty$). If we use linear regression for classification:
* The output can exceed $[0, 1]$, which is mathematically invalid for probability.
* Adding a single outlier data point far away will shift the linear decision boundary significantly, ruining predictions for close inputs.
* The model treats all wrong predictions equally. It cannot heavily penalize a highly confident wrong prediction (e.g. predicting $99\%$ probability of spam for a genuine email).

```
   y (Class)
   ^
 1 |         *   *   *   *
   |       /   <-- Linear line shifts when outliers are added
   |     /
 0 +---*---*---*-------------> Input (x)
       ^ Decision boundary shifts right!
```

## 3. Intuition
Instead of fitting a straight line directly to classes, we compute a linear score (logit):
$$z = w \cdot x + b$$

We then pass this score through the **Sigmoid function**, which squashes any input value into an S-curve bounded between $0$ and $1$. This output is interpreted as the probability of the positive class ($1$).

```
 Probability (y)
   1 |          /--------* (Spam)
     |        /
 0.5 |      /   <-- S-shaped Sigmoid Curve
     |    /
   0 +--*-------            --> Score (z)
```

By applying a threshold (usually $0.5$), we convert the probability into class $0$ or $1$.
