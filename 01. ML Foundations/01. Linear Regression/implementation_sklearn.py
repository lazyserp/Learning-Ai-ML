"""
================================================================================
SIMPLE LINEAR REGRESSION (SCIKIT-LEARN)
================================================================================

Problem:
How do we model simple linear relationships using industry-standard libraries that
handle optimization and parameter calculation out of the box?

Solution:
Use scikit-learn's LinearRegression wrapper, which solves parameters analytically
using Ordinary Least Squares (OLS).

Mathematics:
OLS solves weights using the closed-form Normal Equation:
W = (X^T * X)^(-1) * X^T * Y

Algorithm:
1. Format input data into a 2D array (shape: [num_samples, num_features]) to fit
   scikit-learn's estimator API.
2. Initialize the LinearRegression estimator.
3. Fit the model to calculate parameters.
4. Extract coefficients (weights) and intercept (bias).

Time Complexity:
- O(N * D^2 + D^3) where N is samples, D is features. For simple regression (D=1), it is O(N).

Space Complexity:
- O(N * D) to store input matrices.

Interview Facts:
- scikit-learn's LinearRegression uses SVD-based Ordinary Least Squares,
  meaning it calculates the exact analytical minimum in one step without gradient descent.
- It does not require a learning rate hyperparameter.
"""

# Imports
import numpy as np
from sklearn.linear_model import LinearRegression

# Data
# scikit-learn expects features in a 2D array of shape (samples, features).
# We reshape our 1D array to shape (4, 1) using reshape(-1, 1).
# We do not strictly need scaling for analytical OLS, but keeping it
# consistent with our numpy implementation is good practice.
x_raw = np.array([500.0, 1000.0, 1500.0, 2000.0], dtype=float)
X = (x_raw / 1000.0).reshape(-1, 1)  # Shape: (4, 1)

# Target labels
y = np.array([30.0, 60.0, 90.0, 120.0], dtype=float)

# Model
# We instantiate the LinearRegression model.
model = LinearRegression()

# Loss
# Sklearn handles the loss (MSE minimization) internally during fitting.

# Optimizer
# There is no SGD optimizer here. Sklearn uses LAPACK's OLS solver.

# Training Loop
# The fit method runs the analytical solver in one step.
model.fit(X, y)

# Parameter Extraction
w = model.coef_[0]  # Sklearn stores weights in coef_ array
b = model.intercept_  # Sklearn stores bias in intercept_

print("--- Model Parameters ---")
print(f"Learned Weight (coef_):      {w:.4f}")
print(f"Learned Bias (intercept_):   {b:.4f}")

# Prediction
# We must format query inputs into a 2D array: shape (1, 1).
new_area = np.array([[1.8]])  # Scaled area (1800 sq ft)
predicted_price = model.predict(new_area)[0]

print("\n--- Inference ---")
print(f"Predicted price for 1800 sq ft: {predicted_price:.2f} Lakhs")

# Evaluation
# We evaluate the performance of our model using R^2 score.
r2_score = model.score(X, y)
print(f"Model R^2 Score (1.0 is perfect): {r2_score:.4f}")

# Common Mistakes
# 1. Passing a 1D array to fit(): Sklearn will throw a ValueError.
#    Always reshape 1D input lists using `.reshape(-1, 1)`.
# 2. Applying different scaling during inference: Querying the model with `1800`
#    instead of `1.8` results in completely wrong outputs.
