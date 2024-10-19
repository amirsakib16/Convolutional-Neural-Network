import torch


tensor1 = torch.tensor([12,45,67,89])
tensor2 = torch.tensor([34,78,56,34])
torch.add(tensor1, tensor2)
print("=========================================================================")
print("=========================================================================")

tensor1 = torch.tensor([3,767,54,32])
tensor2 = torch.tensor([9,45,677,23])
torch.mul(tensor1, tensor2)

print("=========================================================================")
print("=========================================================================")

tensor1 = torch.tensor([12,5476,4,57])
tensor2 = torch.tensor([34,78,56,45])
torch.div(tensor1, tensor2)

print("=========================================================================")
print("=========================================================================")

tensor1 = torch.tensor([[2,4,6,8],
                        [2,9,5,3]])
tensor2 = torch.tensor([[1,7,],
                        [6,9],
                        [5,9],
                        [6,12]])
# torch.matmul(): Supports tensors of any dimensionality.
torch.matmul(tensor1, tensor2)

print("=========================================================================")
print("=========================================================================")

tensor1 = torch.tensor([[2,4,6,8],
                        [2,9,5,3]])
tensor2 = torch.tensor([[1,7,],
                        [6,9],
                        [5,9],
                        [6,12]])
torch.mm(tensor1, tensor2)

print("=========================================================================")
print("=========================================================================")

batch1 = torch.tensor([[[1, 2],
                        [3, 4]],

                        [[5, 6],
                        [7, 8]]])

batch2 = torch.tensor([[[9, 10],
                        [11, 12]],

                        [[13, 14],
                        [15, 16]]])

result = torch.bmm(batch1, batch2)

print("Batch Matrix 1:")
print(batch1)

print("Batch Matrix 2:")
print(batch2)

print("Result of Batch Matrix Multiplication:")
print(result)

print("=========================================================================")
print("=========================================================================")

tensor1 = torch.tensor([1, 2, 3])
tensor2 = torch.tensor([4, 5, 6])
result = torch.stack((tensor1, tensor2), dim=0)

print("Stacked Tensor (dim=0):")
print(result)
print("Shape:", result.shape)


tensor1 = torch.tensor([[1, 2],
                        [3, 4]])

tensor2 = torch.tensor([[5, 6],
                        [7, 8]])

result = torch.stack((tensor1, tensor2), dim=0)

print("Stacked Tensor (dim=0):")
print(result)
print("Shape:", result.shape)


print("=========================================================================")
print("=========================================================================")

