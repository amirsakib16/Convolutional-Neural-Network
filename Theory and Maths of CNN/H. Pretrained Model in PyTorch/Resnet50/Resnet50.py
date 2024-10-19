import torch
import torchvision.models as models
from torchsummary import summary  

resnet50 = models.resnet50(pretrained=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet50 = resnet50.to(device)
summary(resnet50, (3, 224, 224))
example_input = torch.randn(1, 3, 224, 224).to(device)
output = resnet50(example_input)

print("Output shape:", output.shape) 
