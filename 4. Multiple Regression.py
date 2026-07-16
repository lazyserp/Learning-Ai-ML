"""
================================================================================
MULTIPLE LINEAR REGRESSION: A PROBLEM-SOLUTION APPROACH
================================================================================

PROBLEM:
A target variable (e.g., house price) is rarely influenced by just a single feature.
How do we model relationships when we have multiple input features (e.g., area AND 
number of bedrooms)?

SOLUTION:
Use Multiple Linear Regression. We extend the simple regression line into a multi-
dimensional plane/hyperplane.
Formula: y = w1*x1 + w2*x2 + ... + wn*xn + b

Vectorized / Matrix Representation:
Instead of looping over each feature, we stack all features into a matrix X,
all weights into a vector W, and calculate predictions using a dot product:
y_pred = X . W + b

--------------------------------------------------------------------------------
SUB-PROBLEMS & SOLUTIONS IN CODE:
1. Feature representation:
   - Problem: How to represent multiple features per sample.
   - Solution: Use a 2D NumPy array of shape (num_samples, num_features).

2. Scaling / Feature Engineering:
   - Problem: Features have different units/scales (e.g. area vs. rooms).
   - Solution: Keep track of scaling parameters and engineering steps (Feature Engineering)
     to transform input features so they are in a uniform scale.
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 1. DATA PREPARATION (2 Features per sample)
# ------------------------------------------------------------------------------
# Features: [Area in sq ft / 1000, Number of Bedrooms]
X = np.array([
    [0.5, 1],
    [1.0, 2],
    [1.5, 3],
    [2.0, 4]
], dtype=float)

# House Prices in Lakhs (Ground Truth labels)
y = np.array([30, 60, 90, 120], dtype=float)


# ------------------------------------------------------------------------------
# 2. MODEL INITIALIZATION
# ------------------------------------------------------------------------------
# We have 2 features, so we need 2 weights (W) and 1 bias (b).
W = np.array([10.0, 5.0])  # Initial weight guesses
b = 5.0                    # Initial bias guess

print("--- Initial Settings ---")
print(f"Feature Matrix X:\n{X}")
print(f"Weights vector W: {W}")
print(f"Bias b:           {b}")


# ------------------------------------------------------------------------------
# 3. VECTORIZED PREDICTION (The Solution to Multi-dimensionality)
# ------------------------------------------------------------------------------
# PROBLEM: How to predict efficiently for all samples without writing nested loops.
# SOLUTION: Use matrix-vector multiplication (dot product).
predictions = np.dot(X, W) + b

print("\n--- Predictions ---")
for i, pred in enumerate(predictions):
    print(f"Sample {i+1} Features {X[i]} -> Predicted Price: {pred:.1f} Lakhs (Actual: {y[i]:.1f})")


# ------------------------------------------------------------------------------
# 4. TRAINING DEMONSTRATION (Single Gradient Descent Step)
# ------------------------------------------------------------------------------
# PROBLEM: How do we compute gradients for multiple weights?
# SOLUTION: Vectorized gradient calculations.
n = len(y)
errors = predictions - y
loss = np.mean(errors ** 2)

# Gradient calculations using matrix operations:
# dW is shape (num_features,), db is a scalar
dW = (2.0 / n) * np.dot(X.T, errors)
db = (2.0 / n) * np.sum(errors)

print(f"\n--- Evaluation & Gradients ---")
print(f"Current Mean Squared Error (Loss): {loss:.4f}")
print(f"Gradient w.r.t W (dW):             {dW}")
print(f"Gradient w.r.t b (db):             {db:.4f}")
