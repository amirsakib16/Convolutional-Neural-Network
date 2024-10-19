import torch
import torch.nn as nn

tanh = nn.Tanh()


input_tensor = torch.tensor([[-1.0, 2.0, 0.0],
                            [3.0, -4.0, 0.5]])

output = tanh(input_tensor)

print("Input Tensor:\n", input_tensor)
print("Output Tensor after Tanh:\n", output)
