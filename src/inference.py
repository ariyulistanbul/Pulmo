import os
import io
import json
import base64
import numpy as np
import torch
from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from .dicom_utils import load_dicom_series_hu, preprocess_slice_2p5d
from .model import build_model, gradcam_target_layers

def _series_cache_key(series_path: str):
    return series_path.replace("\\", "_").replace(":", "").replace("/", "_")

def load_or_build_cache(series_path: str, cache_dir: str = None):
    vol, zpos, meta = load_dicom_series_hu(series_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        fp = os.path.join(cache_dir, _series_cache_key(series_path) + ".npz")
        if not os.path.isfile(fp):
            np.savez_compressed(fp, vol=vol.astype(np.float32), z_positions=zpos.astype(np.float64))
    return vol, zpos, meta

@torch.inference_mode()
def predict_series_slices(model, vol_hu: np.ndarray, device="cpu", batch_size=32):
    Z = vol_hu.shape[0]
    probs = np.zeros((Z,), dtype=np.float32)

    model.eval()
    for start in range(1, Z-1, batch_size):
        end = min(Z-1, start + batch_size)
        xs = []
        idxs = []
        for z in range(start, end):
            x = preprocess_slice_2p5d(vol_hu, z, out_size=256)
            xs.append(x)
            idxs.append(z)
        X = torch.from_numpy(np.stack(xs, axis=0)).to(device)  # (B,3,256,256)
        logits = model(X)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        for i, z in enumerate(idxs):
            probs[z] = float(p[i])
    return probs

def gradcam_overlay_for_slice(model, backbone, vol_hu, slice_index: int, device="cpu"):
    model.eval()

    x = preprocess_slice_2p5d(vol_hu, slice_index, out_size=256)  # (3,256,256) in [0,1]
    X = torch.from_numpy(x[None, ...]).to(device)

    target_layers = gradcam_target_layers(model, backbone)
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [BinaryClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=X, targets=targets)[0]  # (H,W)

    # Use middle slice channel as base image for overlay (or use RGB = 3ch)
    base = x.transpose(1, 2, 0)  # (H,W,3) in [0,1]
    overlay = show_cam_on_image(base, grayscale_cam, use_rgb=True)

    # return overlay as PIL
    return Image.fromarray(overlay)

def pil_to_base64_png(pil_img: Image.Image):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def run_inference(
    series_path: str,
    weights_path: str,
    backbone="resnet18",
    cache_dir="cache",
    slice_index: int | None = None,
    auto_best: bool = True,
    return_base64: bool = False,
    out_dir="outputs"
):
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(backbone=backbone).to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    vol_hu, zpos, meta = load_or_build_cache(series_path, cache_dir=cache_dir)

    probs = predict_series_slices(model, vol_hu, device=device, batch_size=32)
    if auto_best or slice_index is None:
        best_z = int(np.argmax(probs))
    else:
        best_z = int(slice_index)
        best_z = int(np.clip(best_z, 1, vol_hu.shape[0]-2))

    best_prob = float(probs[best_z])

    overlay_img = gradcam_overlay_for_slice(model, backbone, vol_hu, best_z, device=device)
    heatmap_path = os.path.join(out_dir, f"gradcam_z{best_z}.png")
    overlay_img.save(heatmap_path)

    study_id = meta.get("StudyInstanceUID", None) or os.path.basename(series_path)

    finding = {
        "slice_index": best_z,
        "confidence": best_prob,
        "malignancy_prob": best_prob,
        "heatmap_path": heatmap_path,
    }
    if return_base64:
        finding["heatmap_base64_png"] = pil_to_base64_png(overlay_img)

    out = {
        "study_id": study_id,
        "findings": [finding]
    }
    return out, overlay_img, probs
