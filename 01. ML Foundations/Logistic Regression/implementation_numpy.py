"""
================================================================================
LOGISTIC REGRESSION FROM SCRATCH (NUMPY)
================================================================================

Problem:
How do we model binary classification problems (e.g. spam detection) and output
reliable probabilities instead of unbounded values?

Solution:
Apply the Sigmoid function to squash linear combinations into the [0, 1] range,
and optimize parameters using Binary Cross-Entropy Loss.

Mathematics:
- Logit score:         z = X . W + b
- Sigmoid activation:  P = 1 / (1 + exp(-z))
- BCE Loss:            Loss = -(1/n) * sum(y * log(P) + (1-y) * log(1-P))
- Gradient w.r.t W:    dW = (1/n) * X^T . (P - y)
- Gradient w.r.t b:    db = (1/n) * sum(P - y)

Algorithm:
1. Define the sigmoid activation helper.
2. Initialize weights W of shape (num_features,) and bias b.
3. For each training epoch:
   a. Compute linear logit score z.
   b. Pass z through Sigmoid to get probability vector P.
   c. Calculate Cross-Entropy Loss.
   d. Compute gradients of loss.
   e. Update W and b.

Time Complexity:
- Training: O(Epochs * N * D) where N is samples, D is features.
- Inference: O(D) vector-vector dot product + scalar exponentiation.

Space Complexity:
- O(D) auxiliary space.

Interview Facts:
- MSE is not used for Logistic Regression because the Sigmoid activation causes
  the MSE loss function to become non-convex, leading to many local minima.
- Standard cross-entropy gradients are identical in matrix form to linear regression.
"""

# Imports
import numpy as np

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
# Initialize weights vector and bias
W = np.array([0.01, 0.01])
b = 0.0

# Activation Helper
# PROBLEM: Output values must be squashed between 0 and 1.
# SOLUTION: Use the Sigmoid mathematical function.
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Loss
# PROBLEM: Standard distance loss doesn't penalize confident wrong guesses heavily.
# SOLUTION: Calculate Log Loss (Binary Cross-Entropy).
def calculate_bce_loss(y_true, p_pred):
    # Clip predictions to prevent np.log(0) which returns negative infinity
    p_pred = np.clip(p_pred, 1e-15, 1.0 - 1e-15)
    return -np.mean(y_true * np.log(p_pred) + (1.0 - y_true) * np.log(1.0 - p_pred))

# Optimizer
learning_rate = 0.1
epochs = 1000

# Training Loop
n = len(y)
for epoch in range(epochs):
    # Forward Pass: Compute linear logit and squash into probability
    z = np.dot(X, W) + b
    p = sigmoid(z)

    # Evaluate Loss
    loss = calculate_bce_loss(y, p)

    # Backward Pass: Calculate gradients
    dW = (1.0 / n) * np.dot(X.T, (p - y))
    db = (1.0 / n) * np.sum(p - y)

    # Optimization Step
    W = W - learning_rate * dW
    b = b - learning_rate * db

    # Log progress every 100 epochs
    if epoch % 100 == 0:
         print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | W: {W}, b: {b:.4f}")

# Prediction
# We query the model with a new email containing 2 links and 6 spam words.
new_email = np.array([2.0, 6.0])
logit_score = np.dot(new_email, W) + b
predicted_probability = sigmoid(logit_score)
threshold = 0.5
predicted_class = 1 if predicted_probability >= threshold else 0

print("\n--- Inference ---")
print(f"Query Email Features: Links = 2, Spam Words = 6")
print(f"Predicted Probability of Spam: {predicted_probability:.6f} ({predicted_probability * 100:.2f}%)")
print(f"Final Prediction:             {'Spam' if predicted_class == 1 else 'Genuine'}")

# Evaluation
final_logits = np.dot(X, W) + b
final_probabilities = sigmoid(final_logits)
final_loss = calculate_bce_loss(y, final_probabilities)
print(f"\nFinal Model BCE Loss: {final_loss:.6f}")

# Common Mistakes
# 1. Lack of probability clipping: Forgetting to clip predictions before taking
#    the log can cause log(0) resulting in NaN values in loss calculations.
# 2. Using MSE loss: Optimizing Sigmoid + MSE with gradient descent causes
#    vanishing gradient issues when predictions are highly wrong.
