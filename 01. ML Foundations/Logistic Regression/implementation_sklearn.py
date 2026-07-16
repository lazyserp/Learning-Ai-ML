"""
================================================================================
LOGISTIC REGRESSION (SCIKIT-LEARN)
================================================================================

Problem:
How do we deploy robust binary classification models using production-grade libraries
that include built-in regularization and advanced convergence solvers?

Solution:
Use scikit-learn's LogisticRegression estimator, which optimizes classification weights
with built-in regularization (L2 by default) and advanced solvers (e.g., L-BFGS).

Mathematics:
Minimizes regularized loss:
Loss = BCE_Loss(W) + (1 / C) * L2_Penalty(W)
where C is the inverse regularization strength.

Algorithm:
1. Load X and y features/labels.
2. Initialize LogisticRegression wrapper with desired solver and C parameter.
3. Fit the model.
4. Extract coef_ (weights) and intercept_ (bias).
5. Predict class labels or raw class probabilities.

Time Complexity:
- Dependent on solver (e.g., L-BFGS is approximately O(N * D) per iteration).

Space Complexity:
- O(N * D) to store datasets.

Interview Facts:
- predict() returns the discrete class (0 or 1) based on threshold 0.5.
- predict_proba() returns a 2D array containing probabilities for both negative (class 0)
  and positive (class 1) classes.
"""

# Imports
import numpy as np
from sklearn.linear_model import LogisticRegression

# Data
# Features: [Number of Links, Number of Spam Words]
X = np.array([
    [1.0, 5.0],
    [0.0, 1.0],
    [3.0, 8.0],
    [0.0, 0.0]
], dtype=float)

# Target labels (1 = Spam, 0 = Genuine)
y = np.array([1.0, 0.0, 1.0, 0.0], dtype=float)

# Model
# We set C=1e5 (very large C) to disable regularization for comparison
# with our manual NumPy implementation.
model = LogisticRegression(C=1e5, solver='lbfgs')

# Loss
# Minimized internally using regularized cross-entropy.

# Optimizer
# L-BFGS optimization solver.

# Training Loop
# Fit parameters in one call
model.fit(X, y)

# Parameter Extraction
W = model.coef_[0]       # Learned weights: shape (2,)
b = model.intercept_[0]  # Learned bias intercept scalar

print("--- Model Parameters ---")
print(f"Learned Weights (coef_):      {W}")
print(f"Learned Bias (intercept_):   {b:.4f}")

# Prediction
# We query the model for an email containing 2 links and 6 spam words.
# Input must be 2D array.
new_email = np.array([[2.0, 6.0]])
predicted_class = model.predict(new_email)[0]
predicted_probabilities = model.predict_proba(new_email)[0]

print("\n--- Inference ---")
print(f"Predicted Class:         {predicted_class} ({'Spam' if predicted_class == 1 else 'Genuine'})")
print(f"Class 0 (Genuine) Prob:  {predicted_probabilities[0]:.6f}")
print(f"Class 1 (Spam) Prob:     {predicted_probabilities[1]:.6f}")

# Evaluation
accuracy = model.score(X, y)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Common Mistakes
# 1. Confusing predict() and predict_proba(): Calling predict() when you need
#    soft probabilities for ranking or custom thresholding.
# 2. Ignoring regularization: Forgetting that LogisticRegression in scikit-learn
#    applies L2 regularization by default, which biases weights unless C is large.
