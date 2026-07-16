"""
================================================================================
INTRODUCTION TO NEURAL NETWORKS: A PROBLEM-SOLUTION APPROACH
================================================================================

PROBLEM:
As data becomes highly complex (e.g., thousands of pixels in an image, or raw audio
signals), manually designing features (like x^2, x^3, interaction features) 
becomes mathematically impossible and doesn't scale. 

Without manual feature engineering, simple models like linear/logistic regression 
cannot capture these complex non-linear boundaries (lack of MODEL CAPACITY).

SOLUTION:
Neural Networks (NN). 
By stacking multiple layers of mathematical operations and non-linear activation 
functions, Neural Networks possess high MODEL CAPACITY and perform AUTOMATIC 
FEATURE LEARNING. They learn to represent and extract useful features directly from 
raw data.

--------------------------------------------------------------------------------
HOW IT WORKS CONCEPTUALLY (Multi-layer Forward Pass):
1. Input Layer (Raw features)
2. Hidden Layer(s): Combines raw inputs and applies a non-linear activation.
   This step automatically creates new, high-level features!
3. Output Layer: Uses these automatically learned features to make the final prediction.
================================================================================
"""

import numpy as np

# Simple ReLU activation function: max(0, z)
# PROBLEM: Linear transformations combined are still linear.
# SOLUTION: Introduce non-linearity using activation functions like ReLU.
def relu(z):
    return np.maximum(0, z)

# ------------------------------------------------------------------------------
# 1. RAW INPUTS
# ------------------------------------------------------------------------------
# Let's say we have 3 raw features (e.g. pixels of a tiny image)
x = np.array([0.5, -1.2, 0.8])
print("--- Raw Input Features (x) ---")
print(x)


# ------------------------------------------------------------------------------
# 2. HIDDEN LAYER (Automatic Feature Learning)
# ------------------------------------------------------------------------------
# PROBLEM: Raw features cannot be directly mapped to the target with a straight line.
# SOLUTION: Learn a hidden representation (hidden features).
# Weight matrix for Hidden Layer (shape: hidden_neurons, input_features)
# Let's assume 2 hidden neurons (meaning we want to learn 2 new features)
W1 = np.array([
    [0.2, 0.8, -0.5],  # Weights for hidden feature 1
    [-0.6, 0.1, 0.9]   # Weights for hidden feature 2
])
b1 = np.array([0.1, -0.2])  # Biases for hidden layer

# Step A: Linear combination (Logit)
z1 = np.dot(W1, x) + b1

# Step B: Non-linear Activation (Squashing/Thresholding)
# This breaks linearity, allowing the network to learn curved decision boundaries.
h = relu(z1)

print("\n--- Hidden Layer (Learned Features 'h') ---")
print(f"Linear score (z1):        {z1}")
print(f"Activated features (h):   {h}")
print("Note: The network has transformed 3 raw inputs into 2 non-linear features.")


# ------------------------------------------------------------------------------
# 3. OUTPUT LAYER (Final Prediction)
# ------------------------------------------------------------------------------
# We use the learned features (h) to make the final prediction.
W2 = np.array([0.5, -0.3])  # Weights for output
b2 = 0.1                     # Bias for output

y_pred = np.dot(W2, h) + b2

print("\n--- Output Layer ---")
print(f"Predicted Output: {y_pred:.4f}")
print("Summary: NN learned intermediate features 'h' to make the final prediction.")