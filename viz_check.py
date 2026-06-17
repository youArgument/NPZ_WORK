import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "Reff"))

from schemas import DataBatch, SearchParams, TargetData, TargetType, COLOR_MAP_BY_LABEL, NAME_MAP_BY_LABEL
from EntryPoint_temp import _prepare_image, search


def find_npz_files(root_dir: str) -> List[str]:
    npz_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in sorted(filenames):
            if f.endswith(".npz"):
                npz_files.append(os.path.join(dirpath, f))
    return npz_files


def load_batch(npz_path: str) -> Tuple[DataBatch, SearchParams]:
    with np.load(npz_path) as npz:
        magnetogram = npz["magnetogram"]
        velocity = npz["velocity"]
        accelerometer = npz["accelerometer"]
        orientation = npz["orientation"]
        odomstep = float(npz.get("odomstep", 0.002))
        batch = DataBatch(magnetogram, velocity, accelerometer, orientation)
        params = SearchParams(StartDistance=0.0, OdomStep=odomstep)
    return batch, params


def plot_results(
    data_batch: DataBatch,
    target_data: TargetData,
    search_params: SearchParams,
    title: str = "YOLO Detection Results",
    save_path: str = None,
):
    mg = data_batch.Data
    img_arr = _prepare_image(mg, rotation=0)[:, :, 0].T
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.imshow(img_arr, cmap="gray")
    
    if len(target_data.points) == 0:
        ax.text(
            0.5, 0.5, "No detections",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=20, color="red",
        )
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved: {save_path}")
        plt.close(fig)
        return
    
    converted = _plot_coords_from_target(target_data.points, search_params, mg.shape)
    
    # Цвет для всех объектов (можно изменить на любой другой)
    anon_color = "yellow"
    
    for bbox, _ in zip(converted, target_data.labels):
        x, y, w, h = bbox
        w -= x
        h -= y
        rect = patches.Rectangle(
            (x, y), max(w, 1), max(h, 1),
            linewidth=2, edgecolor=anon_color, facecolor=anon_color,
            alpha=0.35, fill=True,
        )
        ax.add_patch(rect)
    
    ax.set_title(
        f"{title}\nDetections: {len(target_data.points)} | "
        f"Shape: {mg.shape[0]}x{mg.shape[1]} | OdomStep: {search_params.OdomStep}",
        fontsize=11,
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def _plot_coords_from_target(points, params, mg_shape):
    out = np.zeros_like(points, dtype=np.float64)
    out[:, 0] = points[:, 0] / params.OdomStep
    out[:, 1] = points[:, 1] * mg_shape[1] / 12.0
    out[:, 2] = points[:, 2] / params.OdomStep
    out[:, 3] = points[:, 3] * mg_shape[1] / 12.0
    return out


def main():
    data_root = os.path.join(os.path.dirname(__file__), "ROW_DATA", "df")
    if not os.path.isdir(data_root):
        print(f"Data root not found: {data_root}")
        return

    npz_files = find_npz_files(data_root)
    if not npz_files:
        print("No .npz files found.")
        return

    print(f"Found {len(npz_files)} .npz files.")

    # files_to_run = npz_files[:5]
    files_to_run = npz_files

    conf = 0.02
    iou = 0.25
    save_dir = os.path.join(os.path.dirname(__file__), "viz_output")
    os.makedirs(save_dir, exist_ok=True)

    for npz_path in files_to_run:
        rel = os.path.relpath(npz_path, data_root)
        print(f"\n{'='*60}")
        print(f"Processing: {rel}")
        print(f"{'='*60}")
    
        batch, params = load_batch(npz_path)
        params.confidence_threshold = conf
        params.iou_threshold = iou
    
        print(f"  Magnetogram shape: {batch.Data.shape}")
        print(f"  Running search (conf={conf}, iou={iou})...")
        result = search(batch, params)
    
        n_dets = len(result.points)
        label_counts = {}
        for l in result.labels:
            label_counts[l] = label_counts.get(l, 0) + 1
        print(f"  Detections: {n_dets}")
        for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"    {NAME_MAP_BY_LABEL.get(lbl, lbl.name)}: {cnt}")
    
        base = os.path.splitext(rel)[0]
        save_path = os.path.join(save_dir, f"{base}_detections.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
        plot_results(
            batch, result, params,
            title=rel.replace("\\", " / "),
            save_path=save_path,
        )

    print(f"\n{'='*60}")
    print(f"Done. Visualizations saved to: {save_dir}")




if __name__ == "__main__":
    main()
