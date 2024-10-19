import torch
import torch.nn as nn

linear_layer = nn.Linear(in_features=3, out_features=2)

input_tensor = torch.tensor([[1.0, 2.0, 3.0]])

output_tensor = linear_layer(input_tensor)

print("Input Tensor:", input_tensor)
print("Output Tensor:", output_tensor)
