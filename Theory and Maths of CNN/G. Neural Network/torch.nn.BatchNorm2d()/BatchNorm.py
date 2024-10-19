import torch
import torch.nn as nn

batch_norm = nn.BatchNorm2d(num_features=64)

input_tensor = torch.randn(1, 64, 32, 32)

output = batch_norm(input_tensor)

print("Input Tensor Shape:", input_tensor.shape)
print("Output Tensor Shape:", output.shape)
