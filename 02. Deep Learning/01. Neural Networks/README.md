# Neural Networks: Multi-Layer Representation

This module covers Multi-Layer Perceptrons (MLPs), the foundation of Deep Learning.

---

## 1. Problem
For complex inputs (like raw image pixels, voice audio, or text sequences), features interact in highly complex, non-linear ways. A model needs to represent curved and multi-dimensional decision boundaries.

## 2. Why Previous Algorithm Failed
Polynomial Regression fits curves by generating combinations of input features (e.g. $x_1^2, x_1 x_2, x_2^2$). However, as the number of features ($d$) grows, the number of polynomial combinations grows exponentially ($O(d^k)$ where $k$ is the degree). This is known as the **Curse of Dimensionality**. 
* E.g., for a tiny $100 \times 100$ pixel image ($10,000$ inputs), a quadratic degree-2 expansion yields $\approx 50$ million inputs, which makes training and generalization impossible.
* Logistic Regression can only fit linear boundaries, unless manual feature engineering is performed.

## 3. Intuition
Instead of manually designing polynomial combinations, a Neural Network stacks layers of weights.
1. The **Hidden Layer** automatically computes new intermediate representations (features) of the data.
2. The **Non-linear Activation** (like ReLU or Sigmoid) distorts the space, allowing subsequent layers to easily draw straight lines through what was previously non-linear data.
3. The **Output Layer** uses these learned intermediate representations to make final predictions.

```
Inputs (Pixels)       Hidden Layer (Learned Features)        Output (Cat/Dog)
   x1  ───\ 
   x2  ─────> [ Linear Combination + Activation ] ─────> Prediction (y_pred)
   x3  ───/
```

By adding more hidden neurons and layers, we increase the model's **Model Capacity** to learn and represent highly complex structures automatically.
