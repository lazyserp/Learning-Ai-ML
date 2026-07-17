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



#my own implementation
import numpy as np

import numpy as np
from typing import Tuple, List

class LogisticRegression:
    """A clean implementation of a Binary Logistic Regression classifier using raw NumPy."""

    def __init__(self, learning_rate: float = 0.1, epochs: int = 3000) -> None:
        """
        Initializes the model hyperparameters and sets placeholders for weights and bias.
        
        Parameters:
            learning_rate: The step size used for gradient descent updates.
            epochs: The number of times the algorithm passes over the dataset.
        """
        self.lr: float = learning_rate
        self.epochs: int = epochs
        self.w: np.ndarray = None  # Weights matrix, initialized dynamically in fit()
        self.b: float = 0.0        # Bias scalar
        self.loss_history: List[float] = []  # Tracks loss over training epochs

    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Applies the sigmoid activation function to map real values into probabilities between 0 and 1.
        
        Parameters:
            z: Input matrix or vector (linear combinations).
        """
        return 1 / (1 + np.exp(-z))

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Performs the forward pass by computing the model's linear activation and probability outputs.
        
        Parameters:
            X: Input feature matrix of shape (num_samples, num_features).
        """
        z = X @ self.w + self.b
        return self.sigmoid(z)

    def compute_loss(self, y: np.ndarray, predictions: np.ndarray) -> float:
        """
        Calculates the Binary Cross-Entropy loss.
        
        Parameters:
            y: Ground truth labels of shape (num_samples, 1).
            predictions: Predicted probabilities of shape (num_samples, 1).
        """
        # Small epsilon prevents log(0) which causes NaN errors
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

    def backward(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Computes the gradients (derivatives) of the loss function with respect to weights and bias.
        
        Parameters:
            X: Input feature matrix of shape (num_samples, num_features).
            y: Ground truth labels of shape (num_samples, 1).
            predictions: Predicted probabilities of shape (num_samples, 1).
        """
        num_samples = len(X)
        error = predictions - y
        dw = (X.T @ error) / num_samples
        db = float(np.mean(error))
        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Trains the logistic regression model using gradient descent.
        
        Parameters:
            X: Training input features of shape (num_samples, num_features).
            y: Training target binary labels of shape (num_samples, 1).
        """
        num_features = X.shape[1]
        self.w = np.zeros((num_features, 1))
        self.b = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            predictions = self.forward(X)
            loss = self.compute_loss(y, predictions)
            self.loss_history.append(loss)
            
            dw, db = self.backward(X, y, predictions)

            # Gradient descent updates
            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.5f}")

        # Final reporting after training completes
        print("\n--- Training Complete ---")
        print(f"Final Learned Weights:\n{self.w}")
        print(f"Final Learned Bias: {self.b:.5f}")
        
        # Calculate training accuracy
        final_preds = self.predict(X)
        accuracy = np.mean(final_preds == y) * 100
        print(f"Training Accuracy: {accuracy:.2f}%\n")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the raw probability estimates for each sample.
        
        Parameters:
            X: Input feature matrix of shape (num_samples, num_features).
        """
        return self.forward(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts hard binary classification labels (0 or 1).
        
        Parameters:
            X: Input feature matrix of shape (num_samples, num_features).
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)
    
    
# X = Hours Studied (6 samples, 1 feature)
X = np.array([[0.5], 
              [1.7], 
              [2.3], 
              [3.5], 
              [4.2], 
              [5.5]])

# y = Pass (1) or Fail (0)
y = np.array([[0], 
              [0], 
              [0], 
              [1], 
              [1], 
              [1]])

model = LogisticRegression()
model.fit(X,y)
print(model.predict(X))