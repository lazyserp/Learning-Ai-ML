"""
================================================================================
NON-LINEAR REGRESSION: A PROBLEM-SOLUTION APPROACH
================================================================================

PROBLEM:
Real-world data is rarely a straight line. If we force a straight line model onto 
curved data, the model is too simple to capture the pattern. This is called 
UNDERFITTING.

SOLUTION:
Perform Feature Engineering to create POLYNOMIAL FEATURES. By raising our existing
input feature to powers (e.g., creating a new feature x^2 or x^3), we can model
curves while still using a linear regression model.

--------------------------------------------------------------------------------
KEY INTERVIEW FACT:
"Linear Regression" is defined by being linear in the PARAMETERS (weights w),
NOT necessarily in the input features (x).
- y = w1*x + w2*(x^2) + b  ==> This is STILL a Linear Regression model because
  the weights (w1, w2) are linear (not squared or inside trigonometric functions).
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 1. UNDERSTANDING THE PROBLEM (Non-Linear Data)
# ------------------------------------------------------------------------------
# Let's say house price grows exponentially or quadratically with size.
x_raw = np.array([1.0, 2.0, 3.0, 4.0])  # Size
y = np.array([2.0, 8.0, 18.0, 32.0])    # Price (Notice: y = 2 * x^2)

# If we try to fit a straight line (y = w * x + b):
# It will fail to fit [1, 2], [2, 8], [3, 18], [4, 32] perfectly because
# the step sizes in y are increasing (6, 10, 14), while x increases linearly.
# This failure is called UNDERFITTING.


# ------------------------------------------------------------------------------
# 2. IMPLEMENTING THE SOLUTION (Polynomial Features)
# ------------------------------------------------------------------------------
# PROBLEM: How do we let a linear model fit this curve?
# SOLUTION: Generate polynomial features (x^2) and add them to our inputs.

# Create the engineered feature: x^2
x_squared = x_raw ** 2

# Stack raw feature and engineered feature to create a new input matrix:
# Columns: [x, x^2]
X_poly = np.stack((x_raw, x_squared), axis=1)

print("--- Polynomial Feature Engineering ---")
print(f"Original Feature (x):        {x_raw}")
print(f"Engineered Feature (x^2):    {x_squared}")
print(f"Combined Feature Matrix X:\n{X_poly}")


# ------------------------------------------------------------------------------
# 3. VERIFICATION
# ------------------------------------------------------------------------------
# If we set weights: w1 = 0 (for x), w2 = 2.0 (for x^2), and bias b = 0
W = np.array([0.0, 2.0])
b = 0.0

# Calculate prediction: y_pred = w1 * x + w2 * x^2 + b
predictions = np.dot(X_poly, W) + b

print("\n--- Predictions using Polynomial Features ---")
print(f"Target values:      {y}")
print(f"Predicted values:   {predictions}")
print("Result: A perfect fit on curved data using a linear equation!")