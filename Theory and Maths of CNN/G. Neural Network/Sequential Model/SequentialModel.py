import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 5),  
    nn.ReLU(),        
    nn.Linear(5, 3)   
)

input_tensor = torch.tensor([[1.0, 2.0, 3.0, 4.0],
                            [4.0, 3.0, 2.0, 1.0]])


output = model(input_tensor)

print("Model architecture:\n", model)
print("\nInput:\n", input_tensor)
print("\nOutput:\n", output)
