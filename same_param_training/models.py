import torch.nn as nn
from torchvision import models
# Model Definition

def our_model(name, freeze_backbone=True):
    """
    (name) -> String -> Name of the Model
    (freeze_backbone) -> Boolean -> Freeze the layers prior to the classifier?
    """

    num_classes = 4

    # FIX: previously these were three independent `if` blocks with no
    # `else`/`raise`. If `name` didn't match any of them, `model` was never
    # assigned and you'd get a confusing `UnboundLocalError` on `return
    # model` instead of a clear message about what went wrong. Converted to
    # if/elif/else with a ValueError on unknown names.

    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        if freeze_backbone:
            model = freeze_resnet(model)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        if freeze_backbone:
            model = freeze_resnet(model)

        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)

    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        if freeze_backbone:
            model = freeze_vgg(model)

        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model-name {name!r}. Supported: 'resnet18', 'resnet50', 'vgg16'."
        )

    return model

def freeze_resnet(model):
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model

def freeze_vgg(model):
    for param in model.features.parameters():
        param.requires_grad = False
    return model