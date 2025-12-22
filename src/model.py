import torch
import torch.nn as nn
import torchvision.models as tvm

def build_model(backbone="resnet18", in_ch=3):
    backbone = backbone.lower()

    if backbone == "resnet18":
        m = tvm.resnet18(weights=None)
        # first conv already expects 3ch, so okay
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    if backbone == "resnet34":
        m = tvm.resnet34(weights=None)
        m.fc = nn.Linear(m.fc.in_features, 1)
        return m

    if backbone == "densenet121":
        m = tvm.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 1)
        return m

    raise ValueError(f"Unknown backbone: {backbone}")

def gradcam_target_layers(model, backbone="resnet18"):
    b = backbone.lower()
    if b.startswith("resnet"):
        return [model.layer4[-1]]
    if b.startswith("densenet"):
        return [model.features.denseblock4]
    # fallback
    return [list(model.modules())[-1]]
