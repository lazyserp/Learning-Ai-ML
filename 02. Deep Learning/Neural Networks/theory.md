# Neural Networks Theory

This document outlines the core concepts of Multi-Layer Perceptrons.

---

## 1. Core Concepts

### Model Capacity
* **Problem**: Simple models lack the ability to represent complex relationships (underfitting).
* **Solution**: Stacking multiple hidden layers of neurons increases the capacity of the model, enabling it to represent arbitrary continuous functions (Universal Approximation Theorem).

### Automatic Feature Learning
* **Problem**: Designing features manually (e.g. edge detectors in computer vision, word pairs in NLP) is expensive and slow.
* **Solution**: Hidden layers learn to represent features automatically during training. For example, in an image classifier:
  - First layer learns to identify edges.
  - Middle layers learn to combine edges into shapes (eyes, noses).
  - Final layers combine shapes into objects (faces).

### Activation Functions
* **Problem**: Stacking linear functions together always results in a linear function (e.g., $W_2(W_1 X + b_1) + b_2 = W_{\text{new}} X + b_{\text{new}}$). Multiple linear layers have no more representation capacity than a single layer.
* **Solution**: Introduce a non-linear activation function (e.g., ReLU, Sigmoid, tanh) between linear steps. This bends the decision boundary, enabling the network to fit non-linear curves.
