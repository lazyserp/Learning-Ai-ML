# Multiple Linear Regression: Stacking Features

This module covers Multiple Linear Regression, which scales linear predictions to multiple inputs.

---

## 1. Problem
Predicting real-world target variables (e.g., house price) is rarely accurate using just a single input feature like area. Additional parameters (e.g., number of bedrooms, age of the house, distance to the city center) all simultaneously influence the final target.

## 2. Why Previous Algorithm Failed
Simple Linear Regression ($y = w \cdot x + b$) only accepts a single feature $x$. If we want to account for multiple inputs, we would have to build multiple isolated models, which:
* Fails to capture interactions between variables (e.g. bedrooms relative to area size).
* Requires manual heuristic consolidation of the different model predictions.
* Increases model complexity and management overhead.

## 3. Intuition
Instead of drawing a straight line in a 2D space, Multiple Linear Regression fits a **flat plane** (for 2 features) or a **hyperplane** (for 3+ features) in a multi-dimensional room.

```
       Price (y)
          ^      /
          |     /  <-- 3D plane fitting our cloud of points
          |   /|/
          |  / /
          | / /
          +--------------------> Size (x1)
         /
        /
       v
   Bedrooms (x2)
```

Each input feature gets its own learnable weight ($w_i$), letting the model weigh the contribution of every input toward the prediction.
