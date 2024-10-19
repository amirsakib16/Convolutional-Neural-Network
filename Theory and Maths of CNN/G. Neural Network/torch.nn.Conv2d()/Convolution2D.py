import torch
import torch.nn as nn


conv_layer = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3)
input_tensor = torch.randn(1, 3, 32, 32)
output = conv_layer(input_tensor)

print(f"Output shape: {output.shape}")