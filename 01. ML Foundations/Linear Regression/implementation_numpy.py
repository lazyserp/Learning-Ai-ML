"""
================================================================================
SIMPLE LINEAR REGRESSION FROM SCRATCH (NUMPY)
================================================================================

Problem:
How do we model the relationship between a single input feature (house area) and 
a continuous target output (house price) without external ML libraries?

Solution:
Initialize weights and bias, scale inputs to maintain optimization stability, 
and iteratively adjust parameters using Gradient Descent to minimize Mean Squared Error.

Mathematics:
- Model Prediction:    y_pred = w * x + b
- Loss (MSE):          Loss = (1/n) * sum((y - y_pred)^2)
- Gradient w.r.t w:    dw = (-2/n) * sum(x * (y - y_pred))
- Gradient w.r.t b:    db = (-2/n) * sum(y - y_pred)
- Parameter Updates:   w = w - learning_rate * dw
                       b = b - learning_rate * db

Algorithm:
1. Divide raw area by 1000 to scale down feature magnitude.
2. Initialize w = 0.01, b = 0.0, and set learning rate = 0.1.
3. For each epoch:
   a. Predict y_pred.
   b. Compute gradients of MSE loss.
   c. Adjust w and b.
4. Predict new house prices by scaling the input query first.

Time Complexity:
- Training: O(Epochs * N) where N is number of samples.
- Inference: O(1) mathematical calculation.

Space Complexity:
- O(1) auxiliary space as we only store scalar weights and bias.

Interview Facts:
- Simple linear regression has a closed-form analytical solution (Normal Equation),
  but gradient descent is preferred for large-scale systems.
- Without feature scaling, gradient descent updates swing wildly, causing divergence.
"""

# Imports
import numpy as np

# Data
# Raw house area in square feet. We scale by 1000 to keep input values in a 
# similar magnitude range as weights, avoiding exploding gradients.
x_raw = np.array([500.0, 1000.0, 1500.0, 2000.0], dtype=float)
x = x_raw / 1000.0  # Feature Scaling: [0.5, 1.0, 1.5, 2.0]

# House prices in Lakhs
y = np.array([30.0, 60.0, 90.0, 120.0], dtype=float)

# Model
# w: parameter representing the slope of the price curve w.r.t size.
# b: parameter representing base price (intercept) when area is 0.
w = 0.01
b = 0.0

# Loss
# Defined as Mean Squared Error (MSE). We square the errors because:
# 1. It makes all errors positive so they don't cancel out.
# 2. It penalizes large errors much more severely than small ones.
def calculate_mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Optimizer
# We use Batch Gradient Descent. We compute the exact derivative of loss
# with respect to parameters over all samples to find the step direction.
learning_rate = 0.1
epochs = 1000

# Training Loop
n = len(x)
for epoch in range(epochs):
    # Forward Pass: Compute predictions based on current parameters
    y_pred = w * x + b

    # Evaluate Loss: Track error reduction
    loss = calculate_mse_loss(y, y_pred)

    # Backward Pass: Calculate gradients (direction of steepest error increase)
    # The -2 multiplier comes from the derivative of the squared loss function.
    dw = (-2.0 / n) * np.sum(x * (y - y_pred))
    db = (-2.0 / n) * np.sum(y - y_pred)

    # Optimization Step: Move w and b opposite to gradient direction
    w = w - learning_rate * dw
    b = b - learning_rate * db

    # Log progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch:03d} | Loss: {loss:.6f} | w: {w:.4f}, b: {b:.4f}")

# Prediction
# We must scale our raw query in the exact same way we scaled training inputs,
# otherwise the learned weight will multiply an unscaled feature (1800 instead of 1.8),
# leading to a massive, invalid prediction.
new_area = 1800.0
scaled_area = new_area / 1000.0
predicted_price = w * scaled_area + b

print("\n--- Inference ---")
print(f"Predicted price for {new_area} sq ft: {predicted_price:.2f} Lakhs")

# Evaluation
# We evaluate the performance of our finalized model.
final_predictions = w * x + b
final_loss = calculate_mse_loss(y, final_predictions)
print(f"Final Model MSE Loss: {final_loss:.6f}")

# Common Mistakes
# 1. Forgetting to scale predictions: Querying the model with `1800` directly
#    instead of scaling to `1.8` leads to predictions that are 1000x too large.
# 2. Setting learning rate too high: Setting learning_rate = 10.0 without scaling
#    causes weight updates to overshoot the minimum, leading to divergence (NaN values).
