"""Проверка доступности CUDA и базового использования YOLO (ultralytics).

Запуск:
  uv run python cuda_check.py --device cuda
  uv run python cuda_check.py --device cpu

На MacBook CUDA обычно отсутствует (будет использоваться CPU/MPS).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--tile-size", type=int, default=256)
    args = p.parse_args()

    import torch

    if args.device == "auto":
        wanted = None
    elif args.device == "cpu":
        wanted = "cpu"
    else:
        wanted = "cuda"

    print("[torch] cuda.is_available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("[torch] cuda device count:", torch.cuda.device_count())
        print("[torch] current device:", torch.cuda.current_device())
        print("[torch] device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    # Проверка через ultralytics YOLO: создаём модель и делаем один predict на синтетическом изображении.
    from ultralytics import YOLO

    repo_root = Path(__file__).resolve().parent
    model_path = repo_root / "models" / "yolo_finetuned.pt"
    if not model_path.exists():
        # fallback на текущую модель в вашем репозитории
        alt = repo_root / "models" / "yolo.pt"
        if not alt.exists():
            print("Model not found:", model_path, "and", alt)
            return 2
        model_path = alt

    print("[yolo] loading:", model_path)
    model = YOLO(str(model_path), task="detect")

    # Синтетическое изображение: (H,W,3) uint8
    H = args.tile_size
    W = args.tile_size
    img = (np.random.rand(H, W) * 255).astype(np.uint8)
    img3 = np.stack([img, img, img], axis=-1)

    predict_kwargs = {
        "conf": 0.01,
        "iou": 0.45,
        "agnostic_nms": True,
        "verbose": False,
        "show": False,
    }
    if wanted == "cuda":
        predict_kwargs["device"] = "cuda"
    elif wanted == "cpu":
        predict_kwargs["device"] = "cpu"

    print("[yolo] predict on synthetic image... device:", predict_kwargs.get("device", "auto"))
    out = model.predict(img3, **predict_kwargs)
    n_boxes = 0
    if out and out[0].boxes is not None:
        n_boxes = len(out[0].boxes)
    print("[yolo] done. boxes:", n_boxes)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
