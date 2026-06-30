import os
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from schemas import DataBatch, SearchParams, SupportData, TargetData, TargetType, COLOR_MAP_BY_LABEL, NAME_MAP_BY_LABEL
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

def get_utils_dir():
    file_path = os.path.abspath(__file__)
    p = Path(file_path)
    dir_path = str(p.parent.parent
                   )
    return dir_path
sys.path.append(get_utils_dir())

from EntryPoint_temp import search


def _build_npz_file_path(npz_datasets_dir: str, batch_idx: int) -> str:
    file_path = os.path.join(npz_datasets_dir, f"data_{batch_idx:05d}.npz")
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"NPZ file not found: {file_path}")
    return file_path


def load_npz_data_batch(npz_datasets_dir: str, batch_idx: int) -> Tuple[DataBatch, SearchParams]:
    """
    Load one data batch from npz_datasets/data_batches by batch index.
    """
    file_path = _build_npz_file_path(npz_datasets_dir, batch_idx)
    with np.load(file_path) as npz_data:
        magnetogram = npz_data['magnetogram']  # shape: [M, N]
        velocity = npz_data['velocity']        # shape: [M]
        accelerometer = npz_data['accelerometer']  # shape: [M, 3]
        orientation = npz_data['orientation']      # shape: [M]
        odomstep = npz_data.get("odomstep", 0.0)

        return (DataBatch(magnetogram, velocity, accelerometer, orientation),
                SearchParams(StartDistance=0.0, OdomStep=float(odomstep)))

def load_npz_data_batch_extra(npz_datasets_dir: str, batch_idx: int) -> Tuple[DataBatch, TargetData, SearchParams]:
    """
    Load one data batch from npz_datasets/data_batches by batch index.
    """
    file_path = _build_npz_file_path(npz_datasets_dir, batch_idx)
    with np.load(file_path) as npz_data:
        magnetogram = npz_data['magnetogram']  # shape: [M, N]
        velocity = npz_data['velocity']        # shape: [M]
        accelerometer = npz_data['accelerometer']  # shape: [M, 3]
        orientation = npz_data['orientation']      # shape: [M]
        odomstep = npz_data.get("odomstep", 0.0)

        points=npz_data["points"].astype(np.float64)
        labels=[TargetType(label) for label in npz_data["labels"]]
        
        return (DataBatch(magnetogram, velocity, accelerometer, orientation),
                TargetData(points, labels),
                SearchParams(StartDistance=0.0, OdomStep=odomstep))

def visualize_annotations_from_npz(
    ax,
    target_batch: TargetData,
    linewidth: float = 2,
    alpha: float = 0.6,
    legend_loc: str = 'upper right',

) -> None:
    patches_list = []
    labels = []
    used_labels = {}
    
    for bbox, label in zip(target_batch.points, target_batch.labels):
        x, y, w, h = bbox
        w -= x
        h -= y 
        
        color = COLOR_MAP_BY_LABEL.get(label, "black")

        rect = patches.Rectangle(
            (x, y), w-1, h,
            linewidth=linewidth,
            edgecolor=color,
            facecolor=color,
            alpha=alpha,
            fill=False
        )
        ax.add_patch(rect)

        labels.append(label.value)
        patches_list.append(rect)
        if label not in used_labels:
            used_labels[label] = color
    legend_handles = [
        patches.Rectangle((0, 0), 1, 1,
                  linewidth=linewidth,
                  edgecolor=color,
                  facecolor=color,
                  alpha=alpha,
                  label=NAME_MAP_BY_LABEL[label])
        for label, color in used_labels.items()
    ]
    ax.legend(handles=legend_handles, loc=legend_loc)

def plot_magnitogram(data_batch: DataBatch, 
                     target_batch: Optional[TargetData] = None,
                     data_params: Optional[SearchParams] = None):
    
    imgarr = data_batch.Data.T
    imgarr = 255 * (imgarr - imgarr.min()) / (imgarr.max() - imgarr.min())
    pil_image = Image.fromarray(imgarr)
    image_np = np.array(pil_image)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np, cmap="binary")

    if target_batch is not None:
        plot_points = np.zeros_like(target_batch.points, dtype=np.float64)
        plot_points[:, 0] = target_batch.points[:, 0] / data_params.OdomStep
        plot_points[:, 1] = target_batch.points[:, 1] * data_batch.Data.shape[1] / 12.0
        plot_points[:, 2] = target_batch.points[:, 2] / data_params.OdomStep
        plot_points[:, 3] = target_batch.points[:, 3] * data_batch.Data.shape[1] / 12.0
        visualize_annotations_from_npz(
            ax=ax,
            target_batch=TargetData(plot_points, target_batch.labels),
        )

    plt.tight_layout()
    plt.show()

    
def plot_magnitogram_normal(
    data_batch: DataBatch,
    target_batch: Optional[TargetData] = None,
    data_params: Optional[SearchParams] = None):
    
    # Отображение может быть независимым от той нормализации,
    # которую использует YOLO в EntryPoint_temp._prepare_image.
    imgarr = data_batch.Data.T

    
    # Мин-макс нормализация только для визуализации.
    lo = float(imgarr.min())
    hi = float(imgarr.max())
    if hi == lo:
        imgarr_u8 = np.zeros_like(imgarr, dtype=np.float32)
    else:
        imgarr_u8 = 255.0 * (imgarr - lo) / (hi - lo)
    

    pil_image = Image.fromarray(imgarr_u8)
    image_np = np.array(pil_image)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(image_np, cmap="binary")

    if target_batch is not None:
        plot_points = np.zeros_like(target_batch.points, dtype=np.float64)
        plot_points[:, 0] = target_batch.points[:, 0] / data_params.OdomStep
        plot_points[:, 1] = target_batch.points[:, 1] * data_batch.Data.shape[1] / 12.0
        plot_points[:, 2] = target_batch.points[:, 2] / data_params.OdomStep
        plot_points[:, 3] = target_batch.points[:, 3] * data_batch.Data.shape[1] / 12.0
        visualize_annotations_from_npz(
            ax=ax,
            target_batch=TargetData(plot_points, target_batch.labels),
        )

    plt.tight_layout()
    plt.show()

def main():
    """
    Minimal example of loading batches by index and calling search.
    """
    # SET YOUR DATA PATH
    npz_datasets_dir = '/mnt/data/infotech_files/1400 2025 Ур-Уж 2025 нарезка NPZ/ИМР/npz_datasets'

    for batch_idx in range(4):
        # Load data & get params
        input_data, params = load_npz_data_batch_extra(npz_datasets_dir, batch_idx)
        # plot_magnitogram(input_data)  # data visualisation

        # Run entry point

        params.confidence_threshold = 0.02
        params.iou_threshold = 0.25 
        params.yolo_rotation = getattr(params, "yolo_rotation", 0)

        # yolo_normalize влияет ТОЛЬКО на вход YOLO.
        yolo_normalize = True

        result = search(
            input_data,
            params,
            normalize=yolo_normalize, # Нормализация перед подачей в модель
            tile_size=512, #Это размер тайла резки - чем меньше, тем больше времени на детекцию так как идёт резка. Чем больше, тем быстрее, но мелкие объекты могут потеряться 
            overlap_ratio=0.75, #Это параметр - Перекрытие, т.е сколько % при резке на тайлы будет добавляться с предыдущего кадра
        )

        # Plot results
        # Нормализованный предпросмотр, оставил исходную функцию plot_magnitogram
        plot_magnitogram_normal(
            input_data,
            result,
            params
        )


if __name__ == "__main__":
    main()
