"""
================================================================================
POLYNOMIAL REGRESSION (SCIKIT-LEARN)
================================================================================

Problem:
How do we efficiently scale polynomial feature engineering and linear fitting using
industrial tools?

Solution:
Use scikit-learn's preprocessing class `PolynomialFeatures` combined with the
`LinearRegression` estimator.

Mathematics:
Normal Equation OLS solver fitted on preprocessed design matrix:
W = (X_poly^T * X_poly)^(-1) * X_poly^T * Y

Algorithm:
1. Reshape base 1D feature X to 2D column vector.
2. Initialize PolynomialFeatures(degree=2, include_bias=False) to generate higher-power columns.
3. Transform input vector X to matrix X_poly using `.fit_transform()`.
4. Fit the LinearRegression estimator on X_poly.

Time Complexity:
- O(N * D) to construct features, O(N * D^2 + D^3) to solve regression,
  where N is samples, D is degree.

Space Complexity:
- O(N * D) to store preprocessed arrays.

Interview Facts:
- Setting `include_bias=True` in PolynomialFeatures adds a column of 1s, representing
  the intercept term directly in the matrix. LinearRegression can handle intercept fitting
  independently if set to False.
"""

# Imports
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Data
# Raw inputs
x_raw = np.array([1.0, 2.0, 3.0, 4.0], dtype=float).reshape(-1, 1)

# Quadratic target outputs
y = np.array([2.0, 8.0, 18.0, 32.0], dtype=float)

# Model
# Define the polynomial expansion step
poly_transformer = PolynomialFeatures(degree=2, include_bias=False)

# Instantiate standard LinearRegression estimator
linear_model = LinearRegression()

# Loss
# Internal MSE minimization during OLS fitting.

# Optimizer
# Analytical solver (LAPACK OLS).

# Training Loop
# Step 1: Preprocess feature vector to construct [x, x^2] matrix
X_poly = poly_transformer.fit_transform(x_raw)

# Step 2: Fit model on preprocessed features
linear_model.fit(X_poly, y)

# Parameter Extraction
w = linear_model.coef_          # Coefficients corresponding to [x, x^2]
b = linear_model.intercept_     # Y-intercept bias term

print("--- Model Parameters ---")
print(f"Learned Coefficients:      {w}")
print(f"Learned Intercept:        {b:.4f}")

# Prediction
# We query the model for a house of size 2.5.
# The query MUST be transformed using the same polynomial transformer first.
query_raw = np.array([[2.5]])
query_poly = poly_transformer.transform(query_raw)
predicted_price = linear_model.predict(query_poly)[0]

print("\n--- Inference ---")
print(f"Raw query:         {query_raw[0][0]}")
print(f"Transformed query: {query_poly[0]}")
print(f"Predicted Output:  {predicted_price:.2f} (Expected: 2 * 2.5^2 = 12.50)")

# Evaluation
r2_score = linear_model.score(X_poly, y)
print(f"Model R^2 Score: {r2_score:.4f}")

# Common Mistakes
# 1. Forgetting to transform the query input during inference: Querying the model
#    with raw `[2.5]` instead of `[2.5, 6.25]` causes ValueError due to feature shape mismatch.
# 2. Re-fitting the transformer on test data: Always call `.transform()` on test/query
#    data using the pre-fitted transformer, never `.fit_transform()`.
