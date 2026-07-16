"""
================================================================================
POLYNOMIAL REGRESSION FROM SCRATCH (NUMPY)
================================================================================

Problem:
How do we model curved data patterns (e.g. accelerating house price trends) using 
standard linear equations without manual rule-writing?

Solution:
Engineer polynomial features (raising input features to powers like x^2) and combine 
them into a design matrix, then compute predictions using linear matrix algebra.

Mathematics:
- Design Matrix (degree=2): X_poly = [x, x^2]
- Forward Prediction:       y_pred = X_poly . W + b
                            y_pred = w1 * x + w2 * (x^2) + b

Algorithm:
1. Load raw 1D feature x.
2. Square the feature values to generate x^2.
3. Stack x and x^2 column-wise to form a 2D feature matrix X_poly of shape (n, 2).
4. Initialize weights W of shape (2,) and bias b.
5. Compute predictions using vectorized matrix dot product.

Time Complexity:
- Feature Transformation: O(N * D) where D is polynomial degree.
- Inference:              O(D) dot product.

Space Complexity:
- O(N * D) to store design matrix.

Interview Facts:
- Polynomial regression is linear in the parameters, not in the inputs.
  Therefore, it is treated mathematically as Multiple Linear Regression.
"""

# Imports
import numpy as np

# Data
# Raw inputs representing size of house
x_raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

# Ground truth outputs (perfectly fit by quadratic equation: y = 2 * x^2)
y = np.array([2.0, 8.0, 18.0, 32.0], dtype=float)

# Model
# We set target coefficients representing w1=0.0 (x coefficient),
# w2=2.0 (x^2 coefficient), and bias b=0.0.
# W shape must match degree: (2,)
W = np.array([0.0, 2.0])
b = 0.0

# Loss
# Standard Mean Squared Error (MSE)
def calculate_mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Optimizer
# We perform analytical verification of the polynomial fit.

# Training Loop
# Instead of iterating, we demonstrate the feature engineering step.
# Step 1: Create higher-power feature x^2
x_squared = x_raw ** 2

# Step 2: Stack features to form the design matrix
# X_poly shape: (4, 2)
X_poly = np.stack((x_raw, x_squared), axis=1)

print("--- Feature Matrix Generation ---")
print(f"Raw Input (x):         {x_raw}")
print(f"Squared Input (x^2):   {x_squared}")
print(f"Design Matrix X_poly:\n{X_poly}")

# Prediction
# Compute predictions using vectorized dot product
y_pred = np.dot(X_poly, W) + b

print("\n--- Inference ---")
print(f"Predictions: {y_pred}")
print(f"Actual:      {y}")

# Evaluation
loss = calculate_mse_loss(y, y_pred)
print(f"MSE Loss on curved data: {loss:.6f} (Perfect Fit)")

# Common Mistakes
# 1. Scaling after polynomial expansion: Scaling features *before* expanding
#    them keeps values in control. Expanding raw numbers like 2000 to 4,000,000
#    and then trying to train leads to numeric overflow.
# 2. Over-specifying degree: Setting degree=10 for 4 data points. This creates
#    artificial oscillations between points, causing massive prediction errors.
