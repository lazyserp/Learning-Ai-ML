"""
================================================================================
MULTIPLE LINEAR REGRESSION (SCIKIT-LEARN)
================================================================================

Problem:
How do we model linear relationships with multiple features using industry-standard
libraries?

Solution:
Use scikit-learn's LinearRegression estimator, which handles arbitrary numbers of
input features out of the box using analytical Ordinary Least Squares solvers.

Mathematics:
Normal Equation solving the weights vector:
W = (X^T * X)^(-1) * X^T * Y

Algorithm:
1. Load features into a 2D array X of shape (num_samples, num_features).
2. Instantiate LinearRegression.
3. Fit the model to calculate coefficients and intercept.

Time Complexity:
- O(N * D^2 + D^3) where N is samples, D is features.

Space Complexity:
- O(N * D) to store input matrices.

Interview Facts:
- coef_ will contain a vector of weights corresponding to each feature column in X.
- LinearRegression automatically handles intercept fitting unless `fit_intercept=False`
  is passed.
"""

# Imports
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
# Columns: [Area scaled, Bedrooms]
X = np.array([
    [0.5, 1.0],
    [1.0, 2.0],
    [1.5, 3.0],
    [2.0, 4.0]
], dtype=float)

# Target prices in Lakhs
y = np.array([30.0, 60.0, 90.0, 120.0], dtype=float)

# Model
model = LinearRegression()

# Loss
# Internal MSE minimization during OLS fitting.

# Optimizer
# Analytical solver (LAPACK OLS).

# Training Loop
# Executed in a single line.
model.fit(X, y)

# Parameter Extraction
W = model.coef_          # Weights vector: shape (2,)
b = model.intercept_     # Bias intercept scalar

print("--- Model Parameters ---")
print(f"Learned Weights (coef_):      {W}")
print(f"Learned Bias (intercept_):   {b:.4f}")

# Prediction
# We query the model with a 2D array for a house of 1800 sq ft and 3 bedrooms.
# Shape must be (1, num_features).
new_house = np.array([[1.8, 3.0]])
predicted_price = model.predict(new_house)[0]

print("\n--- Inference ---")
print(f"Predicted price for 1800 sq ft, 3 bedrooms: {predicted_price:.2f} Lakhs")

# Evaluation
r2_score = model.score(X, y)
print(f"Model R^2 Score: {r2_score:.4f}")

# Common Mistakes
# 1. Shape mismatch during prediction: Querying with a 1D array like `[1.8, 3.0]`
#    instead of 2D array `[[1.8, 3.0]]`. Sklearn expects a batch of inputs.
# 2. Including redundant (perfectly collinear) features: This makes the (X^T * X)
#    matrix singular, causing large numeric instability in weights.
