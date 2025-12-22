import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from .dicom_utils import preprocess_slice_2p5d

class Slice25DDataset(Dataset):
    def __init__(self, items, cache_dir=None, out_size=256, k_context=1, train=True):
        """
        items: list of dict:
          {
            "series_path": str,
            "z": int,
            "y": int (0/1),
            "patient_id": str,
            "series_id": str,
          }
        cache_dir: where .npz volumes stored
        """
        self.items = items
        self.cache_dir = cache_dir
        self.out_size = out_size
        self.k_context = k_context
        self.train = train

    def __len__(self):
        return len(self.items)

    def _load_cached(self, series_path: str):
        if not self.cache_dir:
            return None
        key = series_path.replace("\\", "_").replace(":", "").replace("/", "_")
        fp = os.path.join(self.cache_dir, key + ".npz")
        if not os.path.isfile(fp):
            return None
        data = np.load(fp, allow_pickle=True)
        vol = data["vol"].astype(np.float32)
        return vol

    def __getitem__(self, idx):
        it = self.items[idx]
        series_path = it["series_path"]
        z = int(it["z"])
        y = int(it["y"])

        vol = self._load_cached(series_path)
        if vol is None:
            # fallback: raise to signal you forgot cache in training pipeline
            raise RuntimeError("Volume cache not found. Build cache first in train.py.")

        x = preprocess_slice_2p5d(vol, z, out_size=self.out_size)

        # light aug (demo-safe)
        if self.train:
            if random.random() < 0.5:
                x = x[:, :, ::-1].copy()
            if random.random() < 0.5:
                x = x[:, ::-1, :].copy()

        x_t = torch.from_numpy(x)  # (3,256,256)
        y_t = torch.tensor([y], dtype=torch.float32)

        meta = {
            "patient_id": it.get("patient_id", ""),
            "series_id": it.get("series_id", ""),
            "series_path": series_path,
            "slice_index": z,
        }
        return x_t, y_t, meta
