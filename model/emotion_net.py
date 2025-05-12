import torch
import torch.nn as nn
from torchvision import models

class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        base_model = models.mobilenet_v2(pretrained=True)
        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.features = base_model.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classfier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(1280, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classfier(x)
        return x