import torch


array = [12,3,57,24,15]
Tensor = torch.tensor(array)
print(f"Tensor from an array: \n{Tensor}")

print("=========================================================================")
print("=========================================================================")

emptyTensor = torch.empty(4,3)
print(f"Empty Tensor: \n{emptyTensor}")

print("=========================================================================")
print("=========================================================================")

zeroTensor = torch.zeros(4,3)
print(f"Zero Tensor: \n{zeroTensor}")

print("=========================================================================")
print("=========================================================================")

oneTensor = torch.ones(4,7)
print(f"One Tensor: {oneTensor}")

print("=========================================================================")
print("=========================================================================")

customizedTensor = torch.arange(5,27,3)
print(f"Customized Tensor: {customizedTensor}")

print("=========================================================================")
print("=========================================================================")

linspaceTensor = torch.linspace(7,8,12)
print(f"Linspace Tensor: {linspaceTensor}")

print("=========================================================================")
print("=========================================================================")

randomTensor = torch.rand(3,6)
print(f"Random Tensor: {randomTensor}")

print("=========================================================================")
print("=========================================================================")

randomTensor = torch.randn(3,6)
print(f"Random Tensor: {randomTensor}")

print("=========================================================================")
print("=========================================================================")

eyeTensor = torch.eye(5,5)
print(f"Eye Tensor: {eyeTensor}")

print("=========================================================================")
print("=========================================================================")
