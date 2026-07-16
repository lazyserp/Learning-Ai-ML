"""
================================================================================
MULTIPLE LINEAR REGRESSION FROM SCRATCH (NUMPY)
================================================================================

Problem:
How do we model relationships when predictions depend on multiple input features
(e.g., house size AND number of bedrooms) in a vectorized, loop-free manner?

Solution:
Stack all features into a 2D matrix X, weights into a 1D vector W, and use NumPy's
dot product to calculate predictions and gradients simultaneously.

Mathematics:
- Vectorized Prediction: y_pred = X . W + b
- MSE Loss:              Loss = (1/n) * sum((y - y_pred)^2)
- Gradient w.r.t W:      dW = (2/n) * X^T . (y_pred - y)
- Gradient w.r.t b:      db = (2/n) * sum(y_pred - y)
- Parameter Updates:     W = W - learning_rate * dW
                         b = b - learning_rate * db

Algorithm:
1. Load features into a 2D array X of shape (num_samples, num_features).
2. Initialize weights W as a zero vector of shape (num_features,) and bias b as 0.0.
3. In the training loop:
   a. Compute vectorized predictions.
   b. Calculate prediction errors.
   c. Use matrix transposition (X.T) to distribute error signals back to weights.
   d. Adjust W and b.

Time Complexity:
- Training: O(Epochs * N * D) where N is samples, D is features.
- Inference: O(D) vector-vector dot product.

Space Complexity:
- O(D) to store weight parameters.

Interview Facts:
- Matrix multiplication (dot product) in NumPy leverages low-level BLAS/LAPACK
  libraries, making it orders of magnitude faster than Python `for` loops.
- The shape of dW must match W exactly: shape (D,).
"""

# Imports
import numpy as np

# Data
# Columns: [Area scaled (sq ft / 1000), Number of Bedrooms]
X = np.array([
    [0.5, 1.0],
    [1.0, 2.0],
    [1.5, 3.0],
    [2.0, 4.0]
], dtype=float)

# House prices in Lakhs
y = np.array([30.0, 60.0, 90.0, 120.0], dtype=float)

# Model
# We initialize weights to small values and bias to 0.0.
# W shape must match number of columns in X: (2,)
W = np.array([0.01, 0.01])
b = 0.0

# Loss
def calculate_mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Optimizer
learning_rate = 0.05
epochs = 1000

# Training Loop
n = len(y)
for epoch in range(epochs):
    # Forward Pass: Vectorized matrix multiplication
    # y_pred will have shape (4,) matching y
    y_pred = np.dot(X, W) + b

    # Evaluate Loss
    loss = calculate_mse_loss(y, y_pred)

    # Compute prediction errors
    errors = y_pred - y

    # Backward Pass: Calculate gradients using matrix transposition
    # X.T shape is (2, 4), errors shape is (4,) -> dW shape is (2,)
    dW = (2.0 / n) * np.dot(X.T, errors)
    db = (2.0 / n) * np.sum(errors)

    # Optimization Step: Gradient descent update
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # Log progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | W: {W}, b: {b:.4f}")

# Prediction
# We query the model with a new house of 1800 sq ft and 3 bedrooms.
# Input features must be scaled in the exact same format: [1.8, 3.0]
new_house = np.array([1.8, 3.0])
predicted_price = np.dot(new_house, W) + b

print("\n--- Inference ---")
print(f"Features: Area = 1.8 (1800 sq ft), Bedrooms = 3")
print(f"Predicted Price: {predicted_price:.2f} Lakhs")

# Evaluation
final_predictions = np.dot(X, W) + b
final_loss = calculate_mse_loss(y, final_predictions)
print(f"Final Model MSE Loss: {final_loss:.6f}")

# Common Mistakes
# 1. Dimension Mismatch: Writing `np.dot(W, X)` instead of `np.dot(X, W)`.
#    X has shape (N, D) and W has shape (D,). Dot product requires inner dimensions to match.
# 2. Forgetting to transpose X in gradient step: Trying to dot X directly with errors
#    instead of X.T.
