import torch
import torch.nn as nn

bce_loss = nn.BCELoss()

import torch
import torch.nn as nn


bce_loss = nn.BCELoss()


y_true = torch.tensor([1.0, 0.0, 1.0])  # Actual values (1s and 0s)
y_pred = torch.tensor([0.9, 0.1, 0.8])  # Predicted probabilities

loss = bce_loss(y_pred, y_true)
print("Binary Cross-Entropy Loss:", loss.item())
