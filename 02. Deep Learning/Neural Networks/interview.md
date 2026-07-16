# Neural Networks Interview Questions

This document prepares you for high-frequency questions regarding Neural Networks in technical interviews.

---

## 1. Questions & Answers

### Q. What is the Universal Approximation Theorem, and what is its condition?
* **Answer**: The Universal Approximation Theorem states that a feed-forward neural network with a single hidden layer and a finite number of neurons can approximate any continuous function on a compact subset of $\mathbb{R}^n$ to arbitrary precision. 
* **Condition**: The activation function must be **non-linear**. If the activation function is linear, the model collapses into a simple linear model, losing its approximation power.

### Q. Why is non-linear activation essential in Neural Networks?
* **Answer**: Stacking multiple linear layers yields a linear function:
  $$Y = W_2(W_1 X + b_1) + b_2 = (W_2 W_1) X + (W_2 b_1 + b_2) = W_{\text{new}} X + b_{\text{new}}$$
  Without a non-linear activation between the layers, a deep network is mathematically equivalent to a single-layer linear model. Non-linear activations warp the representation space, enabling the model to learn complex, curved boundaries.

### Q. What is the Curse of Dimensionality, and how does Deep Learning address it?
* **Answer**: The Curse of Dimensionality refers to the exponential growth in volume associated with adding extra dimensions to space. For classical ML (like Polynomial Regression), the number of features explodes exponentially with the degree of interactions. 
* Deep Learning addresses this through **Automatic Feature Learning** and representation reuse. By stacking layers, higher-level features are represented as combinations of lower-level features (e.g., combining edges to form shapes, then objects), avoiding the need to compute all combinations explicitly.

---

## 2. Common Beginner Mistakes
1. **Forgetting activations between layers**: Stacking multiple `nn.Linear` layers directly without `nn.ReLU()` in between, collapsing the network into a simple linear model.
2. **Forgetting `torch.no_grad()` during inference**: Tracking gradients during inference, which consumes unnecessary GPU memory and slows down execution.
3. **Mismatched dimensions**: Defining hidden layers where the output dimension of layer $l$ does not match the input dimension of layer $l+1$.
