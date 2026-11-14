import torch.nn as nn
from torchvision import models
from src.config import num_classes

class MetalDefectResNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(MetalDefectResNet, self).__init__()
    
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class MetalDefectEfficientNet(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(MetalDefectEfficientNet, self).__init__()

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
      
        for param in self.model.parameters():
            param.requires_grad = False
            
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


class MetalDefectEfficientNetUnfreezed(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(MetalDefectEfficientNetUnfreezed, self).__init__()

        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.features[-3:].parameters():
            param.requires_grad = True

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)