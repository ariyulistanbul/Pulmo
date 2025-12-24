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

def gradcam_target_layers(model, backbone: str):
    """
    Works for:
    - Plain torchvision ResNet (model.layer4)
    - Wrapped model having .backbone (model.backbone.layer4)
    - Wrapped model having .encoder (common pattern)
    """

    # 1) wrapper varsa
    if hasattr(model, "backbone"):
        m = model.backbone
    elif hasattr(model, "encoder"):
        m = model.encoder
    else:
        # 2) wrapper yoksa model zaten resnet olabilir
        m = model

    # ResNet family
    if hasattr(m, "layer4"):
        return [m.layer4[-1]]

    # DenseNet gibi alternatifler için (ileride lazım olursa)
    if hasattr(m, "features"):
        # DenseNet'te genelde son denseblock iyi çalışır
        return [m.features[-1]]

    raise ValueError(f"Cannot find target layer for Grad-CAM. Model type: {type(model)}")
