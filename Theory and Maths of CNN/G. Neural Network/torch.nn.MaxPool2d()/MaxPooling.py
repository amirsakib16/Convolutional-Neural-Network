import torch
import torch.nn as nn


max_pool = nn.MaxPool2d(kernel_size=2)

input_tensor = torch.tensor([[[[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [9, 10, 11, 12],
                            [13, 14, 15, 16]]]])

output = max_pool(input_tensor)

print("Input Tensor:")
print(input_tensor)
print("Output Tensor:")
print(output)
