import torch

tensor = torch.tensor([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])

index02 = torch.tensor([0, 2])
index21 = torch.tensor([2,1])

resultA = torch.index_select(tensor, dim=0, index=index02)
resultB = torch.index_select(tensor, dim=0, index=index21)
resultC = torch.index_select(tensor, dim=1, index=index02)
resultD = torch.index_select(tensor, dim=1, index=index21)

print("Selected Rows:")
print(f"Selected dimension row for 0 and 2: \n{resultA}")
print(f"Selected dimension row for 2 and 1: \n{resultB}")
print(f"Selected dimension column for 0 and 2: \n{resultC}")
print(f"Selected dimension column for 2 and 1: \n{resultD}")

tensor_3d = torch.tensor([[[1, 2], [3, 4]],
                        [[5, 6], [7, 8]],
                        [[9, 10], [11, 12]]])

index = torch.tensor([0, 2])
result = torch.index_select(tensor_3d, dim=0, index=index)

print("Selected Elements from 3D Tensor:")
print(result)

print("=========================================================================")
print("=========================================================================")

tensor = torch.tensor([[10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90]])

index = torch.tensor([[0, 2, 1],    
                        [1, 0, 2],    
                        [2, 1, 0]])   

resultX = torch.gather(tensor, dim=0, index=index)
resultY = torch.gather(tensor, dim=1, index=index)
print("Gathered Tensor for dimension 0:")
print(resultX)
print("Gathered Tensor for dimension 1:")
print(resultY)

tensor_3d = torch.tensor([[[10, 20], [30, 40]],
                        [[50, 60], [70, 80]]])

index = torch.tensor([[[0, 1], [1, 0]],
                    [[1, 0], [0, 1]]])

result = torch.gather(tensor_3d, dim=2, index=index)

print("Gathered 3D Tensor:")
print(result)

print("=========================================================================")
print("=========================================================================")

tensor = torch.zeros(3, 3, dtype=torch.int64)

index = torch.tensor([[0, 2, 1],    
                    [0, 1, 2],    
                    [1, 0, 2]])   

src = torch.tensor([[10, 20, 30],   
                    [40, 50, 60],   
                    [70, 80, 90]])  

tensor.scatter_(dim=1, index=index, src=src)

print("Modified Tensor after scatter_:")
print(tensor)

print("=========================================================================")
print("=========================================================================")

tensor = torch.tensor([1, 2, 3, 4, 5, 6])

splittedQ = torch.split(tensor, 2)
print("Splitted by 2")
for i in splittedQ:
    print(i)

tensor = torch.tensor([10, 20, 30, 40, 50])

splittedR = torch.split(tensor, [1, 3, 1])
print("Splitted by 1-3-1")
for j in splittedR:
    print(j)

print("=========================================================================")
print("=========================================================================")

