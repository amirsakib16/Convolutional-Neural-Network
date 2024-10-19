import torch
import torch.nn as nn

mse_loss = nn.MSELoss()
y_true = torch.tensor([2.0, 3.0, 4.0])  # Actual values
y_pred = torch.tensor([2.5, 2.5, 5.0])  # Predicted values

# Calculate MSE loss
loss = mse_loss(y_pred, y_true)
print("Mean Squared Error Loss:", loss.item())  
