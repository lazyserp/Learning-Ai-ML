"""
================================================================================
2-LAYER NEURAL NETWORK FORWARD PASS (NUMPY)
================================================================================

Problem:
How do we compute forward transformations in multi-layer neural networks without
deep learning frameworks?

Solution:
Stack linear combinations (matrix-vector dot products) and insert non-linear
activation functions (like ReLU) between layers to learn complex representations.

Mathematics:
- Hidden Layer Linear combinations:  z1 = W1 . x + b1
- Non-linear Activation (ReLU):    h = max(0, z1)
- Output Layer Linear combination:  y_pred = W2 . h + b2

Algorithm:
1. Load a raw 1D input feature vector x.
2. Multiply by weight matrix W1 and add bias b1 to get hidden logits.
3. Apply ReLU element-wise to set negative logits to 0, representing activation.
4. Multiply hidden representation by weight matrix W2 and add bias b2 to predict final output.

Time Complexity:
- Forward pass: O(H * I + O * H) where I is input size, H is hidden size, O is output size.

Space Complexity:
- O(H) to store intermediate activations.

Interview Facts:
- Without the non-linear activation (ReLU), the operations collapse:
  W2 . (W1 . x + b1) + b2 = (W2.W1) . x + (W2.b1 + b2), which is just a single
  linear layer. Activations are mandatory for representation capacity.
"""

# Imports
import numpy as np

# Data
# A 3-dimensional raw input vector (e.g. 3 pixels)
x = np.array([0.5, -1.2, 0.8])

# Model
# Layer 1 weights (shape: 2 neurons, 3 inputs)
# Layer 1 bias (shape: 2 neurons)
W1 = np.array([
    [0.2, 0.8, -0.5],
    [-0.6, 0.1, 0.9]
])
b1 = np.array([0.1, -0.2])

# Layer 2 weights (shape: 1 output neuron, 2 inputs)
# Layer 2 bias (shape: 1 output neuron)
W2 = np.array([0.5, -0.3])
b2 = 0.1

# Loss
# Standard classification or regression evaluation loss can follow, but
# we focus on the representation forward step.

# Optimizer
# We verify the mathematical transformations of the forward pass.

# Training Loop (Single forward propagation demonstration)
# Step A: Compute logit scores for Hidden Layer 1
z1 = np.dot(W1, x) + b1

# Step B: Apply non-linear activation (ReLU)
# PROBLEM: Keeping the operations linear restricts the network capacity.
# SOLUTION: Set negative values to zero (ReLU activation).
h = np.maximum(0, z1)

# Step C: Compute output logit score
y_pred = np.dot(W2, h) + b2

print("--- Forward Propagation ---")
print(f"Inputs (x):                    {x}")
print(f"Hidden layer logits (z1):      {z1}")
print(f"Hidden layer activations (h):  {h}")
print(f"Predicted Output (y_pred):     {y_pred:.4f}")

# Prediction
# We can interpret y_pred as a regression estimate or feed it into a sigmoid
# for binary prediction.

# Evaluation
# We confirm that shapes align: Input (3,) -> Hidden (2,) -> Output (1,)
print("\n--- Shape Verification ---")
print(f"Input shape:        {x.shape}")
print(f"Weight W1 shape:    {W1.shape}")
print(f"Hidden activation:  {h.shape}")
print(f"Weight W2 shape:    {W2.shape}")

# Common Mistakes
# 1. Missing activation function: Multiplying weights without activations
#    makes the model equivalent to a simple linear regression.
# 2. Shape Mismatch during initialization: Initializing W1 with shape (input, hidden)
#    instead of (hidden, input), which prevents dot products with input x.
