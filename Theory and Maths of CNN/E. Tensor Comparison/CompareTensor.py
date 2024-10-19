import torch


tensorA = torch.tensor([12,45,67,89])
tensorB = torch.tensor([32,67,54,34])
tensorC = torch.tensor([12,45,76,89])

X = torch.eq(tensorA, tensorB)
print(X)
Y = torch.eq(tensorA, tensorC)
print(Y)

print("=========================================================================")
print("=========================================================================")

X = torch.ne(tensorA, tensorB)
print(X)
Y = torch.ne(tensorA, tensorC)
print(Y)

print("=========================================================================")
print("=========================================================================")

X = torch.gt(tensorA, tensorB)
print(X)
Y = torch.gt(tensorA, tensorC)
print(Y)

print("=========================================================================")
print("=========================================================================")
X = torch.lt(tensorA, tensorB)
print(X)
Y = torch.lt(tensorA, tensorC)
print(Y)

print("=========================================================================")
print("=========================================================================")
