
import torch.nn as nn
from torchvision import models
import timm


def get_model(model_name, pretrained=True):

    if model_name == "resnet18":

        weights = (
            models.ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        model = models.resnet18(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

    elif model_name == "resnext50":

        weights = (
            models.ResNeXt50_32X4D_Weights.DEFAULT
            if pretrained else None
        )

        model = models.resnext50_32x4d(weights=weights)

        model.fc = nn.Linear(
            model.fc.in_features,
            2
        )

    elif model_name == "vit_tiny":

        model = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=2,
            img_size=256
        )

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model
