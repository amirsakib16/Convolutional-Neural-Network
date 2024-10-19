import torch


Tensor = torch.tensor([12,45,67,87,65,43,123,45,78])
print(f"Original Tensor: {Tensor}")
viewTensor = Tensor.view(3,3)
print(f"View Tensor: \n{viewTensor}")

print("=========================================================================")
print("=========================================================================")

Tensor = torch.tensor([12,45,67,87,65,43,123,45,78,45,67,89])
print(f"Original Tensor: {Tensor}")
reshapeTensor = Tensor.reshape(3,4)
print(f"Reshape Tensor: \n{reshapeTensor}")

print("=========================================================================")
print("=========================================================================")

tensor1 = torch.tensor([[12,6,78,90,65],[34,567,89,564,234]])
print(f"Original Tensor: \n{tensor1}")
transposeTensor = tensor1.transpose(0,1)
print(f"Transpose Tensor: \n{transposeTensor}")

tensor2 = tensor = torch.rand(2, 3, 4)
print(f"Original Tensor: \n{tensor2}")
transposeTensor = tensor2.transpose(0,2)
print(f"Transpose Tensor: \n{transposeTensor}")

print("=========================================================================")
print("=========================================================================")

tensor = torch.rand(2,4,3)
print(f"Original Tensor: \n{tensor}")
permuteTensor = tensor.permute(2,0,1) # <-- size(3,2,4)
print(f"Permuted Tensor: \n{permuteTensor}")

print("=========================================================================")
print("=========================================================================")

tensor = torch.rand(2,4)
print(f"Shape of the tensor is : {tensor.shape}")
unsqueezeTensor = tensor.unsqueeze(0)
print(f"Shape of the tensor is : {unsqueezeTensor.shape}")
unsqueezeTensor = tensor.unsqueeze(1)
print(f"Shape of the tensor is : {unsqueezeTensor.shape}")
squeezeTensor = tensor.squeeze(0)
print(f"Shape of the tensor is : {squeezeTensor.shape}")

print("=========================================================================")
print("=========================================================================")

