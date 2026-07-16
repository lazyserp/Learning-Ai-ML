"""
================================================================================
MULTI-LAYER PERCEPTRON (PYTORCH)
================================================================================

Problem:
How do we write clean, scalable neural network layers that automatically track
gradients and leverage GPU hardware acceleration for backpropagation?

Solution:
Inherit from PyTorch's `nn.Module` and use built-in linear layers and activation functions.

Mathematics:
PyTorch builds a dynamic computation graph. When performing operations, it stores
inputs and intermediate steps to evaluate gradients automatically via autograd.

Algorithm:
1. Define a class inheriting from `torch.nn.Module`.
2. Initialize linear operations using `nn.Linear(input_dim, output_dim)`.
3. Set forward function computing layers and activations.
4. Pass input tensors through the class instance.

Time Complexity:
- Forward pass: O(H * I + O * H) where I is inputs, H is hidden dimensions, O is outputs.

Space Complexity:
- O(H) to store activation graphs.

Interview Facts:
- nn.Module handles parameter registration automatically; calling `.parameters()`
  yields all learnable weight and bias parameters.
- PyTorch tensors default to float32, whereas NumPy defaults to float64.
"""

# Imports
import torch
import torch.nn as nn

# Data
# PyTorch tensors require explicit type declaration (float32 is standard for DL).
# Input of shape (1, 3) representing 1 sample with 3 features.
x = torch.tensor([[0.5, -1.2, 0.8]], dtype=torch.float32)

# Model
# We build a simple MLP with 1 hidden layer (2 neurons) and 1 output neuron.
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Linear transformation layers
        self.hidden = nn.Linear(in_features=3, out_features=2)
        self.activation = nn.ReLU()
        self.output = nn.Linear(in_features=2, out_features=1)
        
        # We manually overwrite weights to match our NumPy verification values
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[0.2, 0.8, -0.5], [-0.6, 0.1, 0.9]]))
            self.hidden.bias.copy_(torch.tensor([0.1, -0.2]))
            self.output.weight.copy_(torch.tensor([[0.5, -0.3]]))
            self.output.bias.copy_(torch.tensor([0.1]))

    def forward(self, input_tensor):
        # Forward pass: hidden layer -> ReLU activation -> output layer
        z1 = self.hidden(input_tensor)
        h = self.activation(z1)
        y_pred = self.output(h)
        return y_pred

# Initialize class
model = SimpleMLP()

# Loss
# PyTorch provides standard losses like MSELoss.
criterion = nn.MSELoss()

# Optimizer
# We can initialize optimizers like SGD or Adam to update parameters automatically.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training Loop (Single forward pass evaluation)
y_pred = model(x)

print("--- PyTorch Forward Propagation ---")
print(f"Input tensor:      {x}")
print(f"Predicted Output:  {y_pred.item():.4f}")

# Prediction
# Make predictions without tracking gradients to save memory
with torch.no_grad():
    prediction = model(x)
    print(f"Inference output:  {prediction.item():.4f}")

# Evaluation
# Verify model parameters
print("\n--- Model Parameters ---")
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")

# Common Mistakes
# 1. Forgetting `super(SimpleMLP, self).__init__()`: This prevents proper layer
#    registration and throws errors when calling model parameters.
# 2. Running inference with gradient tracking active: Always wrap evaluation code
#    inside `with torch.no_grad():` to avoid wasting memory.
