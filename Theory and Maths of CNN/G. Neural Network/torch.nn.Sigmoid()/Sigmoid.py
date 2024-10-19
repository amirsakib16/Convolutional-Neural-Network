import torch
import torch.nn as nn


sigmoid = nn.Sigmoid()

input_tensor = torch.tensor([[-1.0, 2.0, 0.0],
                            [3.0, -4.0, 0.5]])

output = sigmoid(input_tensor)

print("Input Tensor:\n", input_tensor)
print("Output Tensor after Sigmoid:\n", output)
