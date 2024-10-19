import torch
import torch.nn as nn


relu = nn.ReLU()


input_tensor = torch.tensor([[-1.0, 2.0, -0.5],
                            [3.0, -4.0, 0.0]])

output = relu(input_tensor)

print("Input Tensor:\n", input_tensor)
print("Output Tensor after ReLU:\n", output)
