### MNIST model - CNN기반 

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class network(nn.Module):
#     def __init__(self, z_dim=32):
#         super(network, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
#         self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2(x)))
#         x = x.view(x.size(0), -1)
#         return self.fc1(x)


# class autoencoder(nn.Module):
#     def __init__(self, z_dim=32):
#         super(autoencoder, self).__init__()
#         self.z_dim = z_dim
#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
#         self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)

#         self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
#         self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
#         self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)
        
#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2(x)))
#         x = x.view(x.size(0), -1)
#         return self.fc1(x)
   
#     def decode(self, x):
#         x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)
#         x = F.interpolate(F.leaky_relu(x), scale_factor=2)
#         x = self.deconv1(x)
#         x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2)
#         x = self.deconv2(x)
#         x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2)
#         x = self.deconv3(x)
#         return torch.sigmoid(x)
        

#     def forward(self, x):
#         z = self.encode(x)
#         x_hat = self.decode(z)
#         return x_hat

### MVtec model - CNN

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class network(nn.Module):
#     def __init__(self, z_dim=32):
#         super(network, self).__init__()
#         self.pool = nn.MaxPool2d(2, 2)

#         self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
#         self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         # self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)
#         self.fc1 = nn.Linear(4 * 56 * 56, z_dim, bias=False) # 수정된 부분: 입력 크기를 224x224에 맞게 변경

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2(x)))
#         x = x.view(x.size(0), -1)
#         return self.fc1(x)


# class autoencoder(nn.Module):
#     def __init__(self, z_dim=32):
#         super(autoencoder, self).__init__()
#         self.z_dim = z_dim
#         self.pool = nn.MaxPool2d(2, 2)

#         # Encoder
#         self.conv1 = nn.Conv2d(3, 8, 5, bias=False, padding=2)
#         self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
#         self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         # self.fc1 = nn.Linear(4 * 7 * 7, z_dim, bias=False)
#         self.fc1 = nn.Linear(4 * 56 * 56, z_dim, bias=False) # 수정된 부분: 입력 크기를 224x224에 맞게 변경

#         # Decoder
#         self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2) 
#         self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
#         self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
#         self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
#         self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

#     def encode(self, x):
#         x = self.conv1(x)
#         x = self.pool(F.leaky_relu(self.bn1(x)))
#         x = self.conv2(x)
#         x = self.pool(F.leaky_relu(self.bn2(x)))
#         x = x.view(x.size(0), -1)
#         return self.fc1(x)
   
#     def decode(self, x):
#         x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)  
#         print(x.shape)
#         x = F.interpolate(F.leaky_relu(x), size=(56, 56))  # Resize input tensor
#         x = self.deconv1(x)
#         x = F.interpolate(F.leaky_relu(self.bn3(x)), size=(112, 112))  # Resize input tensor
#         x = self.deconv2(x)
#         x = F.interpolate(F.leaky_relu(self.bn4(x)), size=(224, 224))  # Resize input tensor
#         x = self.deconv3(x)
#         return torch.sigmoid(x)
        

#     def forward(self, x):
#         z = self.encode(x)
#         x_hat = self.decode(z)
#         return x_hat
    
### MVtec model - resnet18

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

class network(nn.Module):
    def __init__(self, z_dim=32):
        super(network, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, z_dim)

    def forward(self, x):
        x = self.resnet(x)
        return x           #


class autoencoder(nn.Module):
    def __init__(self, z_dim=32):
        super(autoencoder, self).__init__()
        self.z_dim = z_dim
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, z_dim)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(z_dim // 16, 4, 5, bias=False, padding=2) 
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=3)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def decode(self, x):
        x = x.view(x.size(0), int(self.z_dim / 16), 4, 4)  
        print(x.shape)
        x = F.interpolate(F.leaky_relu(x), size=(56, 56))  # Resize input tensor
        x = self.deconv1(x)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), size=(112, 112))  # Resize input tensor
        x = self.deconv2(x)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), size=(224, 224))  # Resize input tensor
        x = self.deconv3(x)
        return torch.sigmoid(x)
        

    def forward(self, x):
        z = self.resnet(x)
        x_hat = self.decode(z)
        return x_hat