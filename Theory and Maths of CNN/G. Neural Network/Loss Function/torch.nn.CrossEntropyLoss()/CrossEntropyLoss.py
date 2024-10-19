import torch
import torch.nn as nn


logits = torch.tensor([[5.0, 9.0, 7.0]])

target = torch.tensor([1])  # Class 1 is the correct class
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, target)


print(f"Loss: {loss.item()}")
