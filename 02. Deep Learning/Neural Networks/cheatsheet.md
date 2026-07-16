# Neural Networks Cheatsheet (2-Minute Revision)

---

## 1. Core Workflow

```
Problem: Raw features are too complex for manual feature engineering
   │
   ▼
Approach: Multi-Layer Perceptrons (MLPs)
   │
   ▼
Feature Learning: Hidden Layers learn intermediate representations automatically
   │
   ▼
Line-bending: Non-linear Activation (ReLU: max(0, z)) between layers
   │
   ▼
Calculation: Z = W . A + b  ==>  A = activation(Z)
```

---

## 2. Key Mathematics
* **Forward step**: 
  $$Z^{[l]} = W^{[l]} \cdot A^{[l-1]} + b^{[l]}$$
  $$A^{[l]} = g(Z^{[l]})$$
* **ReLU Activation**: 
  $$g(z) = \max(0, z)$$
* **Sigmoid Activation**: 
  $$g(z) = \frac{1}{1 + e^{-z}}$$

---

## 3. High-Frequency Facts
* **Universal Approximation Theorem**: A single hidden layer with non-linear activation can approximate any continuous function.
* **Curse of Dimensionality**: Exponential feature explosion in polynomial methods, resolved by deep neural networks learning features hierarchically.
* **Linear Collapse**: Stacking linear layers without activations collapses the network into a simple linear model.
