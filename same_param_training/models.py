import torch.nn as nn
from torchvision import models
# Model Definition
def our_model():
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_classes = 4

    for param in resnet50.parameters():
        param.requires_grad = False

    num_features = resnet50.fc.in_features
    resnet50.fc = nn.Linear(num_features, num_classes)
    
    return resnet50
