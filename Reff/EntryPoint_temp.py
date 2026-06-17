import os
import sys
from pathlib import Path
from typing import List, Any, Tuple, Optional
import numpy as np

from ultralytics import YOLO


def get_utils_dir():
    file_path = os.path.abspath(__file__)
    p = Path(file_path)
    dir_path = str(p.parent.parent)
    return dir_path

sys.path.append(get_utils_dir())

from utils import convert_csharp_models
from schemas import DataBatch, SearchParams, TargetData, TargetType

TOTAL_HOURS = 12.0

YOLO_MODEL_PATH = os.path.join(get_utils_dir(), "models", "yolo.pt")

YOLO_TO_TARGET = {
    0: TargetType.unknown,
    1: TargetType.weld,
    2: TargetType.valve,
    3: TargetType.marker,
    5: TargetType.zavarka,
    6: TargetType.vrezka,
}

_model_cache: Optional[YOLO] = None


def get_yolo_model() -> YOLO:
    global _model_cache
    if _model_cache is None:
        _model_cache = YOLO(YOLO_MODEL_PATH, task="detect")
    return _model_cache


def _prepare_image(data: np.ndarray, rotation: int = 0, normalize: bool = True) -> np.ndarray:
    """Подготовка магнитограммы к подаче в YOLO.

    `data` приходит как матрица `[distance, sensors]`.
    Если `normalize=True`, для каждого датчика вычитается его медиана по дистанции
    и значения масштабируются в диапазон `[0, 255]`.

    Дальше изображение при необходимости поворачивается и копируется в 3 канала
    (серый -> RGB). Возвращает массив формы `(H, W, 3)`.
    """
    data = data.astype(np.float32)
    
    if normalize:
        data = data - np.median(data, axis=0, keepdims=True)
        dmin = data.min()
        dmax = data.max()
        if dmax == dmin:
            gray = np.zeros((data.shape[0], data.shape[1]), dtype=np.uint8)
        else:
            norm = (data - dmin) / (dmax - dmin)
            gray = (norm * 255).astype(np.uint8)
    else:
        gray = data.astype(np.uint8)
        
    rot_k = (rotation % 360) // 90
    if rot_k:
        gray = np.rot90(gray, k=rot_k)
    
    rgb = np.stack([gray, gray, gray], axis=-1)
    return rgb


def _invert_rotated_box(
    box: np.ndarray,
    base_shape: Tuple[int, int],
    rotation: int,
) -> Tuple[float, float, float, float]:
    """Перевести bbox YOLO из повернутых координат обратно в `[distance, sensors]`."""
    x1, y1, x2, y2 = box
    h_t, w_t = base_shape
    rot_k = (rotation % 360) // 90

    if rot_k == 0:
        return x1, y1, x2, y2
    if rot_k == 1:
        return min(y1, y2), h_t - max(x1, x2), max(y1, y2), h_t - min(x1, x2)
    if rot_k == 2:
        return w_t - x2, h_t - y2, w_t - x1, h_t - y1
    return w_t - max(y1, y2), min(x1, x2), w_t - min(y1, y2), max(x1, x2)


def _generate_tile_grid(
    height: int, width: int, imgsz: int, overlap_ratio: float
) -> List[Tuple[int, int, int, int]]:
    """Сформировать сетку тайлов с перекрытием.

    Возвращает список окон: `(row_start, col_start, row_end, col_end)`.
    """
    step = max(1, int(imgsz * (1.0 - overlap_ratio)))
    tiles = []
    r = 0
    while r < height:
        c = 0
        while c < width:
            r_end = min(r + imgsz, height)
            c_end = min(c + imgsz, width)
            if r_end - (r if r < height else height - imgsz) < 1:
                r = height
                break
            tiles.append((r, c, r_end, c_end))
            c += step
        r += step
    return tiles


def _pad_tile(tile: np.ndarray, imgsz: int) -> np.ndarray:
    
    """Приводим тайл к размеру `imgsz x imgsz` и формату, ожидаемому YOLO.
    Возвращает массив `uint8` формы `(imgsz, imgsz, 3)`.
    Ultralytics YOLO в данном проекте ожидает "картинку" формата (H, W, 3).
    Поэтому всегда приводим к uint8 и к 3-канальному формату. """
    if tile.ndim == 2:
        tile = np.stack([tile, tile, tile], axis=-1)
    elif tile.ndim != 3:
        raise ValueError(f"Unexpected tile ndim={tile.ndim}, expected 2 or 3")

    """ (H, W, C) """
    h, w, c = tile.shape
    if c != 3:
        """Если по какой-то причине пришло не 3 канала — приводим к 3 """
        if c == 1:
            tile = np.repeat(tile, 3, axis=-1)
        else:
            raise ValueError(f"Unexpected tile channels c={c}, expected 3")

    tile_u8 = np.clip(tile, 0, 255).astype(np.uint8, copy=False)

    padded = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    padded[:h, :w] = tile_u8
    return padded


def _prepare_tile_for_yolo(tile: np.ndarray, tile_size: int) -> np.ndarray:
    """Привести tile к формату, который YOLO ожидает для image input.
       Ожидаем HWC, 3 канала."""
    if tile.ndim == 2:
        tile = np.stack([tile, tile, tile], axis=-1)

    if tile.ndim != 3 or tile.shape[2] != 3:
        raise ValueError(f"Unexpected tile shape for YOLO: {tile.shape}. Expected (H,W,3).")

    """ Кадр должен быть uint8."""
    if tile.dtype != np.uint8:
        tile = np.clip(tile, 0, 255).astype(np.uint8, copy=False)

    """ Делаем нужный размер."""
    if tile.shape[0] != tile_size or tile.shape[1] != tile_size:
        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
        hh = min(tile.shape[0], tile_size)
        ww = min(tile.shape[1], tile_size)
        padded[:hh, :ww] = tile[:hh, :ww]
        tile = padded

    return tile


def _compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Считаем IOU для двух bbox в формате `[x1, y1, x2, y2]`."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union == 0:
        return 0.0
    return inter / union


def _deduplicate_boxes(
    boxes: np.ndarray, confs: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Удаляем дубликаты bbox по IOU через greedy NMS.
    Используется для объединения результатов, полученных на пересекающихся тайлах."""
    if len(boxes) == 0:
        return np.array([], dtype=np.int64)

    order = np.argsort(-confs)
    keep = []
    suppressed = [False] * len(boxes)

    while order.size > 0:
        i = order[0]
        keep.append(i)
        suppressed[i] = True
        order = order[1:]

        new_order = []
        for j in order:
            if suppressed[j]:
                continue
            iou = _compute_iou(boxes[i], boxes[j])
            if iou <= iou_threshold:
                new_order.append(j)
            else:
                suppressed[j] = True

        order = np.array(new_order, dtype=np.int64) if new_order else np.array([], dtype=np.int64)

    return np.array(keep, dtype=np.int64)


def convert_points_to_plot_coordinate(
    postprocessed_data: Any,
    batch_shape: Tuple[int, int],
    searchParams: SearchParams,
    transposed_for_yolo: bool = False,
) -> np.ndarray:
    """Перевести bbox из пикселей `[x1, y1, x2, y2]` в координаты `TargetData`.
    `TargetData` хранит точки как `[dist1, hours1, dist2, hours2]`.
    """
    def convert_points(points):
        res = np.zeros_like(points, dtype=np.float64)
        if transposed_for_yolo:
            """YOLO видел data.T: x соответствует исходной строке (distance), y - исходному столбцу (hours)."""
            res[:, 0] = points[:, 0] * searchParams.OdomStep
            res[:, 2] = points[:, 2] * searchParams.OdomStep
            res[:, 1] = points[:, 1] * TOTAL_HOURS / batch_shape[1]
            res[:, 3] = points[:, 3] * TOTAL_HOURS / batch_shape[1]
        else:
            """Исходная ориентация изображения: y - distance, x - hours. """
            res[:, 0] = points[:, 1] * searchParams.OdomStep
            res[:, 2] = points[:, 3] * searchParams.OdomStep
            res[:, 1] = points[:, 0] * TOTAL_HOURS / batch_shape[1]
            res[:, 3] = points[:, 2] * TOTAL_HOURS / batch_shape[1]
        return res

    converted_points = convert_points(postprocessed_data)
    return np.array(converted_points)


def search(data: DataBatch, searchParams: SearchParams, normalize: bool = True, tile_size: int = 640, overlap_ratio: float = 0.85) -> TargetData:
    """Запускаем YOLO-детекцию на магнитограмме.

    Поддерживается режим подготовки входа `normalize`:
    - `True`: вычитание медианы + min-max масштабирование в `[0, 255]`
    - `False`: без нормализации (просто перевод в `uint8`)

    Для больших изображений включен тайлинговый проход: изображение режется
    на пересекающиеся тайлы, детекции из тайлов объединяются и дедуплицируются
    по IOU.
    """

    acp = data.Data
    
    try:
        model = get_yolo_model()
        conf_thresh = searchParams.confidence_threshold
        iou_thresh = searchParams.iou_threshold
        rotation = getattr(searchParams, "yolo_rotation", 0)
    
        img = _prepare_image(acp, rotation=rotation, normalize=normalize)
        """ Убедимся, что изображение для YOLO имеет правильный формат."""
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"_prepare_image must return (H,W,3), got shape={img.shape}")
        img = np.clip(img, 0, 255).astype(np.uint8, copy=False)
        H, W = img.shape[:2]
        base_shape = acp.shape
    
        """ Важно: координаты bbox из ultralytics в разделе ниже мы транслируем в том же формате, 
        что и в исходной реализации (bx = box[:4], cls_id = box[5]).
        Поэтому не используем отдельные conf/cls извлечения, чтобы не сломать интерпретацию формата."""
        results_with_data = []
        tile_grid = _generate_tile_grid(H, W, tile_size, overlap_ratio)
    
        for r1, c1, r2, c2 in tile_grid:
            tile = img[r1:r2, c1:c2]
            padded_tile = _pad_tile(tile, tile_size)
            padded_tile = _prepare_tile_for_yolo(padded_tile, tile_size)
    
            res = model.predict(
                padded_tile,
                conf=conf_thresh,
                iou=iou_thresh,
                agnostic_nms=True,
                verbose=False,
                show=False,
            )
    
            if res and res[0].boxes is not None:
                raw = res[0].boxes.data.cpu().numpy()  # shape: [N, >=6]
                for box in raw:
                    cls_id = int(box[5])
                    # Формат bbox: [x1, y1, x2, y2, conf, cls]
                    x1_rel, y1_rel, x2_rel, y2_rel = box[:4]
                    conf = float(box[4])

                    # Convert tile-local coords to rotated-image coords
                    x1_img = x1_rel + c1
                    y1_img = y1_rel + r1
                    x2_img = x2_rel + c1
                    y2_img = y2_rel + r1

                    # Map rotated-image coords back to original data coords
                    base_box = _invert_rotated_box(
                        (x1_img, y1_img, x2_img, y2_img),
                        base_shape,
                        rotation,
                    )
                    results_with_data.append({
                        'box': base_box,
                        'conf': conf,
                        'cls': cls_id,
                    })
    
        if not results_with_data:
            blank_boxes = np.array([[0, 0, 0, 0]], dtype=float)
            blank_labels = [TargetType.unknown]
            return TargetData(
                points=convert_points_to_plot_coordinate(
                    postprocessed_data=blank_boxes,
                    batch_shape=acp.shape,
                    searchParams=searchParams,
                ),
                labels=blank_labels,
            )
    
        boxes_array = np.array([r['box'] for r in results_with_data], dtype=np.float64)
        confs_array = np.array([r['conf'] for r in results_with_data])
        cls_ids = np.array([r['cls'] for r in results_with_data])
    
        kept_indices = _deduplicate_boxes(boxes_array, confs_array, iou_thresh)
    
        final_boxes = boxes_array[kept_indices]
        final_cls_ids = cls_ids[kept_indices]
    
        final_labels = [YOLO_TO_TARGET.get(int(cid), TargetType.unknown) for cid in final_cls_ids]
    
        return TargetData(
            points=convert_points_to_plot_coordinate(
                postprocessed_data=final_boxes,
                batch_shape=acp.shape,
                searchParams=searchParams,
            ),
            labels=final_labels,
        )
    
    except BaseException as e:
        blank_boxes = np.array([[0, 0, 0, 0]], dtype=float)
        blank_labels = [TargetType.unknown]
        return TargetData(
            points=convert_points_to_plot_coordinate(
                postprocessed_data=blank_boxes,
                batch_shape=acp.shape,
                searchParams=searchParams,
            ),
            labels=blank_labels,
        )
