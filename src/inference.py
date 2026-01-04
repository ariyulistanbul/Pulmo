# =========================
# INDEX CONTRACT (BUNU SAKLA)
# =========================
# slice_probs[i]  <->  dicom_files_sorted[i]  <->  cam_z{i}.png
# UI hangi slice'ı gösteriyorsa (i), her şey o i üzerinden okunacak.
# Sıralama asla "dosya adına göre" varsayılmayacak; inference'ın ürettiği dicom_files_sorted listesi referans.

import os
import json
import numpy as np
import torch
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from .dicom_utils import load_dicom_series_hu
from .model import build_model


# -------------------------
# Helpers (IO / contract)
# -------------------------

def _series_cache_key(series_path: str) -> str:
    return series_path.replace("\\", "_").replace(":", "").replace("/", "_")


def _sorted_dicom_files_by_inference(series_dir: str) -> List[str]:
    """
    IMPORTANT:
    UI'nin sırası = inference sırası olsun diye,
    DICOM sıralamasını load_dicom_series_hu içindeki mantıkla AYNI yapıyoruz.
    load_dicom_series_hu zaten dosyaları okuyup InstanceNumber ağırlıklı sıralıyor.
    Burada sadece "o sıralı fp listesini" geri çıkarmak için küçük bir tekrar yapıyoruz.
    """
    import pydicom

    def _safe_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    dcm_files: List[str] = []
    for root, _, files in os.walk(series_dir):
        for f in files:
            if f.lower().endswith(".dcm"):
                dcm_files.append(os.path.join(root, f))
    if not dcm_files:
        raise FileNotFoundError(f"No DICOM files found in: {series_dir}")

    headers: List[Tuple[str, int, Optional[float]]] = []
    for fp in dcm_files:
        try:
            ds = pydicom.dcmread(fp, force=True, stop_before_pixels=True)
            inst = _safe_int(getattr(ds, "InstanceNumber", 0), 0)
            ipp = getattr(ds, "ImagePositionPatient", None)
            z = None
            if ipp is not None and len(ipp) >= 3:
                try:
                    z = float(ipp[2])
                except Exception:
                    z = None
            if z is None:
                sl = getattr(ds, "SliceLocation", None)
                try:
                    z = float(sl) if sl is not None else None
                except Exception:
                    z = None
            headers.append((fp, inst, z))
        except Exception:
            continue

    headers.sort(key=lambda t: (t[1], 0 if t[2] is None else t[2]))
    return [fp for fp, _, _ in headers]


def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def _write_json(path: str, obj: Dict[str, Any]):
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Core inference utilities
# -------------------------

def _make_25d_triplet(vol_hu: np.ndarray, i: int) -> np.ndarray:
    """
    2.5D: [i-1, i, i+1] stack.
    Borderlarda clip uygulanır.
    Returns: (3, H, W) float32
    """
    Z = int(vol_hu.shape[0])
    i0 = int(np.clip(i - 1, 0, Z - 1))
    i1 = int(np.clip(i, 0, Z - 1))
    i2 = int(np.clip(i + 1, 0, Z - 1))
    x = np.stack([vol_hu[i0], vol_hu[i1], vol_hu[i2]], axis=0).astype(np.float32)
    return x


def _normalize_hu(x: np.ndarray, hu_min: float = -1000.0, hu_max: float = 400.0) -> np.ndarray:
    """
    Baseline HU window normalize -> [0,1]
    x: (3,H,W) or (H,W)
    """
    x = np.clip(x, hu_min, hu_max)
    x = (x - hu_min) / (hu_max - hu_min + 1e-8)
    return x.astype(np.float32)


@torch.inference_mode()
def predict_series_slices(
    model: torch.nn.Module,
    vol_hu: np.ndarray,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """
    Slice-level malignancy prob üretir.
    vol_hu: (Z,H,W) HU volume
    Returns: probs (Z,) float32
    """
    model.eval()
    Z = int(vol_hu.shape[0])

    probs = np.zeros((Z,), dtype=np.float32)

    xs: List[np.ndarray] = []
    idxs: List[int] = []

    for i in range(Z):
        x = _make_25d_triplet(vol_hu, i)
        x = _normalize_hu(x)  # (3,H,W) in [0,1]
        xs.append(x)
        idxs.append(i)

        if len(xs) >= batch_size or i == Z - 1:
            xb = np.stack(xs, axis=0)  # (B,3,H,W)
            xt = torch.from_numpy(xb).to(device=device, dtype=torch.float32)

            # Model output shape'ine göre olası durumları normalize edelim
            out = model(xt)

            # out: (B,) veya (B,1) veya (B,2) gibi olabilir
            if isinstance(out, (tuple, list)):
                out = out[0]

            if out.ndim == 2 and out.shape[1] == 1:
                logits = out[:, 0]
                pb = torch.sigmoid(logits)
            elif out.ndim == 2 and out.shape[1] == 2:
                # binary softmax varsayımı: malign sınıfı index=1
                pb = torch.softmax(out, dim=1)[:, 1]
            else:
                # (B,) gibi
                pb = torch.sigmoid(out.reshape(-1))

            pb_np = pb.detach().cpu().numpy().astype(np.float32)
            for j, ii in enumerate(idxs):
                probs[ii] = float(pb_np[j])

            xs, idxs = [], []

    return probs


def _find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Grad-CAM için bir conv layer bulmak için basit heuristic.
    ResNet türlerinde layer4[-1].conv2 gibi yakalanır genelde.
    Heuristic: model içindeki son Conv2d modülünü döndür.
    """
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last


def gradcam_overlay_for_slice(
    model: torch.nn.Module,
    backbone: str,
    vol_hu: np.ndarray,
    slice_index: int,
    device: str = "cpu",
    out_size: int = 512,
    image_weight: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      overlay_img_u8: (H,W,3) uint8 (opsiyonel kullanırsın)
      cam01: (H,W) float in [0,1]
    """
    model.eval()

    idx = int(slice_index)
    x = _make_25d_triplet(vol_hu, idx)
    x01 = _normalize_hu(x)  # (3,H,W) [0,1]
    xt = torch.from_numpy(x01[None, ...]).to(device=device, dtype=torch.float32)  # (1,3,H,W)

    # Target layer
    target_layer = _find_last_conv_layer(model)

    activations = []
    gradients = []

    def _fwd_hook(_m, _inp, out):
        activations.append(out)

    def _bwd_hook(_m, _gin, gout):
        gradients.append(gout[0])

    h1 = target_layer.register_forward_hook(_fwd_hook)
    h2 = target_layer.register_full_backward_hook(_bwd_hook)

    # Forward
    out = model(xt)
    if isinstance(out, (tuple, list)):
        out = out[0]

    # Skor seçimi (binary varsayımlar)
    if out.ndim == 2 and out.shape[1] == 2:
        score = out[:, 1].sum()
    elif out.ndim == 2 and out.shape[1] == 1:
        score = out[:, 0].sum()
    else:
        score = out.reshape(-1).sum()

    # Backward (Grad-CAM gradient)
    model.zero_grad(set_to_none=True)
    score.backward(retain_graph=False)

    h1.remove()
    h2.remove()

    if not activations or not gradients:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    A = activations[0]  # (1,C,h,w)
    G = gradients[0]    # (1,C,h,w)

    # weights: global-average-pool gradients over spatial dims
    w = G.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (w * A).sum(dim=1, keepdim=True)  # (1,1,h,w)
    cam = torch.relu(cam)

    # normalize to [0,1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    cam01 = cam[0, 0].detach().cpu().numpy().astype(np.float32)  # (h,w)

    # resize to out_size (square) or keep aspect? burada square basit
    cam_img = Image.fromarray((cam01 * 255).astype(np.uint8)).resize((out_size, out_size))
    cam01_rs = (np.asarray(cam_img).astype(np.float32) / 255.0).clip(0, 1)

    # base slice image (middle channel) resize
    base = (x01[1] * 255.0).astype(np.uint8)  # (H,W)
    base_img = Image.fromarray(base).resize((out_size, out_size))
    base_u8 = np.asarray(base_img)

    # simple red overlay (same as UI)
    base_rgb = np.stack([base_u8, base_u8, base_u8], axis=-1).astype(np.float32) / 255.0
    heat = np.zeros_like(base_rgb, dtype=np.float32)
    heat[..., 0] = cam01_rs  # red

    out_rgb = (1 - image_weight) * base_rgb + image_weight * heat
    overlay_u8 = (np.clip(out_rgb, 0, 1) * 255).astype(np.uint8)

    return overlay_u8, cam01_rs


# -------------------------
# Main: Inference -> JSON contract
# -------------------------

@torch.inference_mode()
def run_inference(
    series_dir: str,
    weights_path: str,
    backbone: str = "resnet18",
    out_dir: str = "outputs",
    json_name: str = "result.json",
    cam_mode: str = "on_demand",   # "on_demand" | "precomputed"
    cam_dirname: str = "cam",      # out_dir/<series_key>/cam/
    cam_pattern: str = "cam_z{index}.png",
    precompute_cam: bool = False,  # cam_mode="precomputed" iken otomatik üretir
    device: Optional[str] = None,
    batch_size: int = 32,
) -> str:
    """
    Produces:
      - outputs/<series_key>/result.json
      - optionally outputs/<series_key>/cam/cam_z{index}.png (precomputed)
    Returns:
      path to written JSON
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    # 1) Model yükle
    model = build_model(backbone=backbone).to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # 2) DICOM vol + meta
    vol_hu, zpos, meta = load_dicom_series_hu(series_dir)
    Z = int(vol_hu.shape[0])
    if Z < 3:
        raise RuntimeError(f"Series has too few slices (Z={Z}). Need at least 3 for 2.5D.")

    # 3) DICOM dosyalarını inference sırası ile çıkar (UI index contract için)
    dicom_files_sorted = _sorted_dicom_files_by_inference(series_dir)
    if len(dicom_files_sorted) != Z:
        raise RuntimeError(
            f"DICOM file count ({len(dicom_files_sorted)}) != volume slices ({Z}). "
            f"Some DICOMs may have failed to read. Fix series or adjust loader to keep alignment."
        )

    # 4) probs üret
    probs = predict_series_slices(model, vol_hu, device=device, batch_size=batch_size).astype(np.float32)
    best_idx = int(np.argmax(probs))
    best_prob = float(probs[best_idx])

    # 5) JSON contract
    series_key = _series_cache_key(series_dir)
    series_out_dir = os.path.join(out_dir, series_key)
    _ensure_dir(series_out_dir)

    cam_dir = os.path.join(series_out_dir, cam_dirname)
    if cam_mode not in ["on_demand", "precomputed"]:
        raise ValueError("cam_mode must be 'on_demand' or 'precomputed'.")

    result: Dict[str, Any] = {
        "schema_version": "1.0",
        "series_path": os.path.abspath(series_dir),
        "study_instance_uid": meta.get("StudyInstanceUID", None),
        "series_instance_uid": meta.get("SeriesInstanceUID", None),

        # core contract
        "dicom_files_sorted": [os.path.abspath(p) for p in dicom_files_sorted],
        "slice_probs": [float(x) for x in probs.tolist()],
        "num_slices": int(Z),

        # navigation
        "best_slice_index": best_idx,
        "best_prob": best_prob,

        # CAM contract
        "cam_mode": cam_mode,
        "cam_cache_dir": os.path.abspath(cam_dir),
        "cam_pattern": cam_pattern,

        # model info (nice-to-have)
        "model_backbone": backbone,
        "weights_path": os.path.abspath(weights_path),
    }

    json_path = os.path.join(series_out_dir, json_name)
    _write_json(json_path, result)

    # 6) İstersen precompute CAM
    if cam_mode == "precomputed" and precompute_cam:
        _ensure_dir(cam_dir)
        for i in range(Z):
            if i == 0 or i == Z - 1:
                continue
            get_or_build_cam_for_index(
                json_path=json_path,
                index=i,
                device=device,
                out_size=512,
                image_weight=0.35,
            )

    return json_path


# -------------------------
# On-demand CAM builder (UI toggle)
# -------------------------

def get_or_build_cam_for_index(
    json_path: str,
    index: int,
    device: Optional[str] = None,
    out_size: int = 512,
    image_weight: float = 0.35,
) -> str:
    """
    UI'daki toggle ile çağrılır.
    JSON contract üzerinden cache path + model config okunur.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = _load_json(json_path)

    series_dir = data["series_path"]
    weights_path = data["weights_path"]
    backbone = data.get("model_backbone", "resnet18")
    cam_dir = data["cam_cache_dir"]
    cam_pattern = data.get("cam_pattern", "cam_z{index}.png")

    _ensure_dir(cam_dir)

    Z = int(data["num_slices"])
    idx = int(np.clip(int(index), 1, Z - 2))
    cam_path = os.path.join(cam_dir, cam_pattern.format(index=idx))

    if os.path.isfile(cam_path):
        return cam_path

    # Model yükle (Grad-CAM için)
    model = build_model(backbone=backbone).to(device)
    sd = torch.load(weights_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()

    # Vol yükle
    vol_hu, _zpos, _meta = load_dicom_series_hu(series_dir)

    # Grad-CAM için grad aç
    with torch.enable_grad():
        _overlay_img, cam01 = gradcam_overlay_for_slice(
            model=model,
            backbone=backbone,
            vol_hu=vol_hu,
            slice_index=idx,
            device=device,
            out_size=out_size,
            image_weight=image_weight,
        )

    Image.fromarray((np.clip(cam01, 0, 1) * 255).astype(np.uint8)).save(cam_path)
    return cam_path
