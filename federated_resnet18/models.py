import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4


def build_resnet18(freeze_backbone: bool = True) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, NUM_CLASSES)

    return model