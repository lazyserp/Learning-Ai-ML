"""
================================================================================
LOGISTIC REGRESSION: A PROBLEM-SOLUTION APPROACH
================================================================================

PROBLEM:
How do we predict discrete categories (e.g., Spam [1] vs. Not Spam [0]) rather 
than continuous values, and how do we measure the model's confidence in these 
predictions?

SOLUTION:
Use Logistic Regression. We compute a linear score (logit), pass it through the 
Sigmoid function to squash it into a probability between 0 and 1, and make a 
prediction based on a threshold (usually 0.5). We train the model using 
Binary Cross-Entropy Loss to heavily penalize confident incorrect predictions.

--------------------------------------------------------------------------------
SUB-PROBLEMS & SOLUTIONS IN CONCEPT & CODE:

1. Unbounded Output:
   - Problem: The raw linear score (logit) z = w*x + b can be any real number
     (-infinity to +infinity), which cannot represent a probability.
   - Solution: Pass the logit through the Sigmoid function:
     Sigmoid(z) = 1 / (1 + e^-z)
     This squashes any value into the range [0, 1].

2. Converting Probabilities to Classes:
   - Problem: A probability is continuous (e.g., 0.73), but we need a discrete class (0 or 1).
   - Solution: Apply a Threshold (e.g., if probability >= 0.5, class = 1; else 0).

3. Measuring Classification Error:
   - Problem: Simple distance metrics (like |y_actual - y_pred|) don't penalize 
     catastrophic, high-confidence mistakes heavily enough. For example, 
     predicting 0.9999 probability of Spam when actual label is 0 (Not Spam) 
     is a severe mistake that should receive an near-infinite penalty.
   - Solution: Use Binary Cross-Entropy Loss (Log Loss):
     Loss = - [ y * log(p) + (1 - y) * log(1 - p) ]
     - When y = 1: Loss = -log(p). If p -> 0 (confident wrong), loss -> infinity.
     - When y = 0: Loss = -log(1 - p). If p -> 1 (confident wrong), loss -> infinity.
================================================================================
"""

import numpy as np

# ------------------------------------------------------------------------------
# 1. THE SIGMOID FUNCTION
# ------------------------------------------------------------------------------
# PROBLEM: Squash unbounded scores (logits) into [0, 1] range.
# SOLUTION: Sigmoid(z) = 1 / (1 + exp(-z))
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------------------
# 2. LOGISTIC REGRESSION FORWARD PASS
# ------------------------------------------------------------------------------
# Sample Inputs: x1 (number of links), x2 (number of spam words)
x = np.array([3.0, 5.0])
w = np.array([0.5, 2.0])
b = -1.0

# Step A: Compute the Logit (Linear score)
# z = w1*x1 + w2*x2 + b
z = np.dot(w, x) + b

# Step B: Squash into probability using Sigmoid
probability = sigmoid(z)

# Step C: Thresholding to get class
threshold = 0.5
prediction = 1 if probability >= threshold else 0

print("--- Forward Pass Demonstration ---")
print(f"Features:                 {x}")
print(f"Logit (z):               {z:.4f}")
print(f"Predicted Probability (p): {probability:.6f} ({probability * 100:.3f}%)")
print(f"Threshold:                {threshold}")
print(f"Final Class Prediction:   {prediction}")


# ------------------------------------------------------------------------------
# 3. BINARY CROSS-ENTROPY LOSS (Why confidence matters)
# ------------------------------------------------------------------------------
# PROBLEM: Penalize wrong predictions based on confidence.
# SOLUTION: Log Loss: -[y*log(p) + (1-y)*log(1-p)]
def compute_binary_cross_entropy(y_actual, p_pred):
    # Clip predictions to prevent log(0) which is undefined/negative infinity
    p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)
    return -(y_actual * np.log(p_pred) + (1.0 - y_actual) * np.log(1.0 - p_pred))

print("\n--- Why Binary Cross-Entropy is Crucial ---")

# Scenario A: True class is 0 (Not Spam). The model makes a near-correct prediction.
y_actual_A = 0.0
p_pred_A = 0.51  # Model is uncertain but slightly wrong
loss_A = compute_binary_cross_entropy(y_actual_A, p_pred_A)
print(f"Scenario A (Unconfident Wrong): Actual = {y_actual_A}, Pred Prob = {p_pred_A:.2f}")
print(f" -> Loss: {loss_A:.6f} (Small penalty)")

# Scenario B: True class is 0 (Not Spam). The model is extremely confident and wrong.
y_actual_B = 0.0
p_pred_B = 0.999999  # Model is highly confident but completely wrong!
loss_B = compute_binary_cross_entropy(y_actual_B, p_pred_B)
print(f"Scenario B (Confident Wrong):   Actual = {y_actual_B}, Pred Prob = {p_pred_B:.6f}")
print(f" -> Loss: {loss_B:.6f} (Enormous penalty!)")
print(f"Ratio of Loss (B / A):          {loss_B / loss_A:.2f}x penalty increase")
