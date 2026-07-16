# Linear Regression: Modeling Continuous Outputs

This module covers Simple Linear Regression, which models relationships using a straight line.

---

## 1. Problem
We need to predict a continuous numerical value (e.g., the market price of a house) based on a single input feature (e.g., its square footage).

## 2. Why Previous Approach Failed
In traditional programming, we try to hardcode heuristics (e.g., `price = area * 50`). This approach fails because:
* Real estate markets are dynamic and depend on hidden factors.
* Hand-crafting exact numbers for different cities or neighbourhoods is impossible to maintain.
* Hardcoded formulas do not adapt as new sales data is collected.

## 3. Intuition
Imagine plotting historical house sales on a graph where the horizontal axis represents size and the vertical axis represents price. The points will likely trend upwards. 

Our goal is to draw a single straight line through this cloud of points that sits as close to all points as possible.

```
 Price (y)
   ^
   |           * (Actual Sale)
   |         /
   |       /   <-- Our Linear Regression Line (y = w*x + b)
   |     /  *
   |   /
   | *
   +-----------------------> Size (x)
```

By finding the optimal slope ($w$) and height ($b$) of this line, we can feed in any new house size and read the corresponding price off the line.
