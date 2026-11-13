import torch
import torch.nn as nn
from src.config import num_classes

class MetalDefectCNN(nn.Module):
    def __init__(self, num_classes=num_classes):
        super(MetalDefectCNN, self).__init__()
        
        self.features = nn.Sequential(
            
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))
        )
      
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.5), 
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    dummy_input = torch.randn(4, 1, 500, 1024)
    model = MetalDefectCNN(num_classes=num_classes)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")