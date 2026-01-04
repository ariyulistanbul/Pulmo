import os
import json
import numpy as np
import gradio as gr
from PIL import Image
import pydicom

# inference modülü
from src.inference import run_inference, get_or_build_cam_for_index


DEFAULT_WEIGHTS = os.environ.get("WEIGHTS_PATH", "outputs/best_resnet18.pt")
DEFAULT_BACKBONE = os.environ.get("BACKBONE", "resnet18")


def _load_json(p: str):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _dicom_to_uint8(ds, hu_min=-1000.0, hu_max=400.0):
    """UI'da 'doktor bakışı' için windowing (baseline)."""
    arr = ds.pixel_array.astype(np.int16)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = arr.astype(np.float32) * slope + intercept

    x = np.clip(hu, hu_min, hu_max)
    x = (x - hu_min) / (hu_max - hu_min + 1e-8)
    x = (x * 255.0).astype(np.uint8)
    return x


def _overlay_cam_on_gray(gray_u8: np.ndarray, cam_u8: np.ndarray, alpha: float = 0.45):
    """
    Minimal overlay: cam'ı kırmızı kanal gibi basıyoruz (doktor demo için yeterli).
    UI ekibi daha sonra kendi viewer'ına taşır.
    """
    H, W = gray_u8.shape
    base = np.stack([gray_u8, gray_u8, gray_u8], axis=-1).astype(np.float32)

    cam = cam_u8.astype(np.float32) / 255.0
    heat = np.zeros((H, W, 3), dtype=np.float32)
    heat[..., 0] = cam  # red

    out = (1 - alpha) * (base / 255.0) + alpha * heat
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out


def _read_slice_image(dicom_path: str):
    ds = pydicom.dcmread(dicom_path, force=True)
    gray = _dicom_to_uint8(ds)  # (H,W)
    return gray


def do_inference(series_dir: str):
    if not series_dir or not os.path.isdir(series_dir):
        raise gr.Error("Geçerli bir DICOM seri klasörü seçmelisin (içinde .dcm olmalı).")

    if not os.path.isfile(DEFAULT_WEIGHTS):
        raise gr.Error(f"Model weights bulunamadı: {DEFAULT_WEIGHTS}")

    # Normal inference: JSON contract üretilir (dicom_files_sorted, slice_probs, best_slice_index, vb.)
    json_path = run_inference(
        series_dir=series_dir,
        weights_path=DEFAULT_WEIGHTS,
        backbone=DEFAULT_BACKBONE,
        out_dir="outputs",
        cam_mode="on_demand",
    )

    data = _load_json(json_path)

    # İlk gösterim: best slice
    best_i = int(data["best_slice_index"])
    dicom_path = data["dicom_files_sorted"][best_i]
    gray = _read_slice_image(dicom_path)
    img = Image.fromarray(gray)

    # Full JSON'u da göster
    data["_debug_json_path"] = json_path
    return json_path, best_i, img, float(data["slice_probs"][best_i]), data


def view_slice(json_path: str, index: int, show_cam: bool, opacity: float):
    # JSON yoksa: hata fırlatma, UI'yi boş bırak
    if (not json_path) or (not os.path.isfile(json_path)):
        return None, None

    data = _load_json(json_path)
    Z = int(data["num_slices"])

    i = int(np.clip(int(index), 0, Z - 1))
    dicom_path = data["dicom_files_sorted"][i]
    gray = _read_slice_image(dicom_path)
    prob = float(data["slice_probs"][i])

    # CAM OFF veya border slice'lar (2.5D kısıtı)
    if not show_cam or i == 0 or i == Z - 1:
        return Image.fromarray(gray), prob

    # CAM ON (on-demand cache)
    cam_path = get_or_build_cam_for_index(json_path, index=i)
    cam_u8 = np.asarray(Image.open(cam_path).convert("L"))

    # cam boyutu farklıysa resize
    if cam_u8.shape != gray.shape:
        cam_u8 = np.asarray(
            Image.fromarray(cam_u8).resize((gray.shape[1], gray.shape[0]))
        )

    over = _overlay_cam_on_gray(gray, cam_u8, alpha=float(opacity))
    return Image.fromarray(over), prob


with gr.Blocks() as demo:
    gr.Markdown("# DICOM Viewer + Slice Probability + Grad-CAM Toggle (Local)")

    with gr.Row():
        series_dir = gr.Textbox(
            label="DICOM Series Folder Path",
            placeholder=r"D:\...\series_folder"
        )
        run_btn = gr.Button("Run inference (build JSON)")

    with gr.Row():
        json_path_out = gr.Textbox(label="Result JSON Path", interactive=False)
        slice_idx = gr.Slider(minimum=0, maximum=400, value=0, step=1, label="Slice index")

    with gr.Row():
        cam_toggle = gr.Checkbox(value=False, label="Grad-CAM ON/OFF")
        opacity = gr.Slider(0.0, 1.0, value=0.45, step=0.05, label="CAM opacity")

    with gr.Row():
        img = gr.Image(label="DICOM Slice (with optional CAM)", type="pil")
        prob = gr.Number(label="Malignancy probability (this slice)")

    meta = gr.JSON(label="Run summary")

    # Run inference sets JSON + jumps to best slice
    run_btn.click(
        fn=do_inference,
        inputs=[series_dir],
        outputs=[json_path_out, slice_idx, img, prob, meta],
    )

    # Viewing updates on slider/toggle/opacity
    slice_idx.change(
        fn=view_slice,
        inputs=[json_path_out, slice_idx, cam_toggle, opacity],
        outputs=[img, prob],
    )
    cam_toggle.change(
        fn=view_slice,
        inputs=[json_path_out, slice_idx, cam_toggle, opacity],
        outputs=[img, prob],
    )
    opacity.change(
        fn=view_slice,
        inputs=[json_path_out, slice_idx, cam_toggle, opacity],
        outputs=[img, prob],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
