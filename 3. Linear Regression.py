"""
================================================================================
LINEAR REGRESSION: A PROBLEM-SOLUTION APPROACH
================================================================================

PROBLEM:
How do we model the relationship between a single continuous input feature 
(e.g., house area in sq ft) and a continuous output value (e.g., house price)?

SOLUTION:
Use Linear Regression (y = w * x + b). We train the model using Mean Squared Error
(MSE) loss and update the parameters (weight w and bias b) using Gradient Descent.

--------------------------------------------------------------------------------
SUB-PROBLEMS & SOLUTIONS IN CODE:
1. Feature Scale Mismatch:
   - Problem: Raw feature values (e.g. 500 to 2000 sq ft) are much larger than
     our weights, which makes Gradient Descent highly unstable (divergence).
   - Solution: Scale features to a standard range (e.g., dividing by 1000).

2. Learning Shifts:
   - Problem: A pure multiplier (y = w * x) can only model lines passing through
     the origin. Real data might have a base/offset value even when x is 0.
   - Solution: Add a Bias (b) term (y = w * x + b).

3. Weight Adjustments:
   - Problem: Finding how to update weight (w) and bias (b) to reduce error.
   - Solution: Compute the gradients (partial derivatives of loss w.r.t w and b) 
     and step in the opposite direction.
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 1. DATA PREPARATION
# ------------------------------------------------------------------------------
# SUB-PROBLEM: Raw inputs are too large, leading to gradient explosion.
# SOLUTION: Scale the feature by dividing by 1000.
x_raw = np.array([500, 1000, 1500, 2000], dtype=float)
x = x_raw / 1000.0  # Scaled inputs: [0.5, 1.0, 1.5, 2.0]

# House Prices in Lakhs (Ground Truth labels)
y = np.array([30, 60, 90, 120], dtype=float)


# ------------------------------------------------------------------------------
# 2. MODEL HYPERPARAMETERS & INITIALIZATION
# ------------------------------------------------------------------------------
# Initialize base value (bias) and feature multiplier (weight)
w = 0.01
b = 0.0

learning_rate = 0.1
epochs = 1000


# ------------------------------------------------------------------------------
# 3. THE TRAINING LOOP
# ------------------------------------------------------------------------------
# PROBLEM: How to iteratively adjust weights until they fit the pattern.
# SOLUTION: Repeatedly predict, calculate MSE loss, compute gradients, and update.
n = len(x)

for epoch in range(epochs):
    # Predict: y_pred = w * x + b
    y_pred = w * x + b

    # Compute Loss: Mean Squared Error (MSE)
    loss = np.mean((y - y_pred) ** 2)

    # Compute Gradients (Direction of steepest increase in loss)
    # Derivative of MSE loss w.r.t w: (-2/n) * sum(x * (y - y_pred))
    # Derivative of MSE loss w.r.t b: (-2/n) * sum(y - y_pred)
    dw = (-2.0 / n) * np.sum(x * (y - y_pred))
    db = (-2.0 / n) * np.sum(y - y_pred)

    # Update weights in the opposite direction of the gradient
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Loss (MSE) = {loss:.6f} | w = {w:.4f}, b = {b:.4f}")

print("\n--- Training Complete ---")
print(f"Learned Weight (w): {w:.4f}")
print(f"Learned Bias (b):   {b:.4f}")


# ------------------------------------------------------------------------------
# 4. PREDICTION FOR NEW DATA
# ------------------------------------------------------------------------------
# SUB-PROBLEM: Make a prediction for a new house of 1800 sq ft.
# CAUTION: The input MUST be scaled in the exact same way as training data!
# SOLUTION: Scale the query (1800 / 1000) before applying the formula.
new_area = 1800
scaled_area = new_area / 1000.0
price = w * scaled_area + b

print(f"\nPrediction for new house ({new_area} sq ft):")
print(f"Formula: Price = {w:.4f} * {scaled_area} + {b:.4f}")
print(f"Predicted Price: {price:.2f} Lakhs")
