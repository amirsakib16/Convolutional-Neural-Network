import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image


from torchvision.models import MobileNet_V2_Weights
model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
model.eval()  


transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),           
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
])

img = Image.open('H. Pretrained Model in PyTorch\Mobilenet_v2\INPUT.jpg').convert('RGB')  
img_t = transform(img)  
input_tensor = img_t.unsqueeze(0)  

with torch.no_grad(): 
    output = model(input_tensor) 
    _, predicted = torch.max(output, 1) 


print(f'Predicted class index: {predicted.item()}')

