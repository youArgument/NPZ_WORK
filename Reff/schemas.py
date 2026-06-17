from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

def _as_int16_view(arr, field_name, *, strict_zero_copy=True, type=np.int16):
    """
    Try to build zero-copy int16 view for C# short[].
    Falls back to asarray if buffer protocol is unavailable.
    """
    try:
        view = memoryview(arr)
        np_arr = np.frombuffer(view, dtype=type)
    except TypeError:
        if strict_zero_copy:
            raise ValueError(
                f"{field_name}: zero-copy conversion unavailable (no buffer protocol)"
            )
        # Some interop layers don't expose buffer protocol.
        np_arr = np.asarray(arr, dtype=type)

    if np_arr.dtype != type:
        if strict_zero_copy:
            raise ValueError(
                f"{field_name}: dtype is {np_arr.dtype}, expected int16 for zero-copy conversion"
            )
        np_arr = np_arr.astype(type, copy=False)

    if np_arr.ndim != 1:
        raise ValueError(f"{field_name} must be 1D short[]")
    return np_arr

def _as_float64_view(arr, field_name, *, strict_zero_copy=True):
    try:
        view = memoryview(arr)
        np_arr = np.frombuffer(view, dtype=np.float64)
    except TypeError:
        if strict_zero_copy:
            raise ValueError(f"{field_name}: zero-copy conversion unavailable")
        np_arr = np.asarray(arr, dtype=np.float64)
    if np_arr.ndim != 1:
        raise ValueError(f"{field_name} must be 1D double[]")
    return np_arr

class TargetType(Enum):
    unknown = -1
    weld = 1
    valve = 2
    marker = 3
    zavarka = 5
    vrezka = 6

COLOR_MAP_BY_LABEL = {
        TargetType.unknown: 'yellow',
        TargetType.weld: 'red',
        TargetType.valve: 'blue',
        TargetType.marker: 'green',
        TargetType.zavarka: 'orange',
        TargetType.vrezka: 'purple',
    }
NAME_MAP_BY_LABEL = {
    TargetType.unknown: 'Unknown',
    TargetType.weld: 'Сварной шов',
    TargetType.valve: 'Кран',
    TargetType.marker: 'Маркер',
    TargetType.zavarka: 'Заварка',
    TargetType.vrezka: 'Врезка',
}


class BaseSharpModel(ABC):

    @classmethod
    @abstractmethod
    def from_csharp(cls, params):
        """Build model instance from CSHARP object."""
        raise NotImplementedError

@dataclass
class DataBatch(BaseSharpModel):
    """
    EFD бэйзд моделька.
    """
    Data: np.ndarray      # shape: [M, N]
    Velocity: np.ndarray  # shape: [M]
    Accelerometer: np.ndarray  # shape: [M, 3]
    Orientation: np.ndarray  # shape: [M]
    
    @staticmethod
    def _validate_shapes(data):
        m = len(data.Speed)
        if m == 0:
            raise ValueError("Speed must not be empty")

        for name, arr in (
            ("ToolRotateAngle", data.ToolRotateAngle),
            ("AxelX", data.AxelX),
            ("AxelY", data.AxelY),
            ("AxelZ", data.AxelZ),
        ):
            arr_len = len(arr)
            if arr_len != m:
                raise ValueError(f"{name} length {arr_len} does not match Speed length {m}")

        total = len(data.Data)
        if total % m != 0:
            raise ValueError(
                f"Data length {total} is not divisible by Speed length {m}; "
                "cannot infer matrix shape [M, N]"
            )
    
    @classmethod
    def from_csharp(cls, data):
        """Convert C# object with raw fields into DataBatch instance WITH zero-copy views."""
        cls._validate_shapes(data)
        velocity = _as_int16_view(data.Speed, "Speed", strict_zero_copy=True)
        orientation = _as_int16_view(data.ToolRotateAngle, "ToolRotateAngle", strict_zero_copy=True)
        axel_x = _as_int16_view(data.AxelX, "AxelX", strict_zero_copy=True)
        axel_y = _as_int16_view(data.AxelY, "AxelY", strict_zero_copy=True)
        axel_z = _as_int16_view(data.AxelZ, "AxelZ", strict_zero_copy=True)
        raw_data = _as_int16_view(data.Data, "Data", strict_zero_copy=True)

        m = velocity.shape[0]
        total = raw_data.shape[0]
        n = total // m
        data_matrix = raw_data.reshape(m, n)
        accelerometer = np.column_stack((axel_x, axel_y, axel_z))

        return cls(
            Data=data_matrix,
            Velocity=velocity,
            Accelerometer=accelerometer,
            Orientation=orientation,
        )

@dataclass
class SearchParams(BaseSharpModel):
    """
    Базовые параметры для поиска. Вписывать доп параметры сюда! 
    Значение параметров по умолчанию прописывать здесь.
    """
    StartDistance: float  # DONT TOUCH
    OdomStep: float  # DONT OTUCH
    confidence_threshold: float = 0.3
    iou_threshold: float = 0.25
    imgsz: int = 640
    tile_overlap_ratio: float = 0.2
    yolo_rotation: int = 0

    @classmethod
    def from_csharp(cls, params):
        StartDistance = float(params.StartDistance)
        OdomStep = float(params.OdomStep)
        confidence_threshold = getattr(params, "ConfidenceThreshold", 0.02)
        iou_threshold = getattr(params, "IouThreshold", 0.45)
        imgsz = getattr(params, "Imgsz", 640)
        tile_overlap_ratio = getattr(params, "TileOverlapRatio", 0.2)
        yolo_rotation = getattr(params, "YoloRotation", 0)
        return cls(
            StartDistance=StartDistance,
            OdomStep=OdomStep,
            confidence_threshold=float(confidence_threshold),
            iou_threshold=float(iou_threshold),
            imgsz=int(imgsz),
            tile_overlap_ratio=float(tile_overlap_ratio),
            yolo_rotation=int(yolo_rotation),
        )

@dataclass
class TargetData(BaseSharpModel):
    """
    Целевая модель с разметкой для обучения.
    """
    points: np.ndarray         # shape: [K, 4]
    labels: List[TargetType]   # shape: [K]

    @staticmethod
    def _validate_shapes(data):
        points_total = len(data.Points)
        if points_total == 0:
            raise ValueError("Points must not be empty")
        if points_total % 4 != 0:
            raise ValueError(
                f"Points length {points_total} is not divisible by 4; expected flattened [M, 4]"
            )

        m = points_total // 4
        labels_len = len(data.Labels)
        if labels_len != m:
            raise ValueError(f"Labels length {labels_len} does not match points rows {m}")
    
    @classmethod
    def from_csharp(cls, params):
        cls._validate_shapes(params)
        flat_points = _as_float64_view(params.Points, "Points", strict_zero_copy=True)
        points = flat_points.reshape(-1, 4)
        labels = [TargetType(label) for label in params.Labels]
        return cls(points=points, labels=labels)

@dataclass
class SupportData(BaseSharpModel):
    """
    Дополнительная информация для обучения, которая можен быть передана.
    """
    thicks: np.ndarray  # shape: [M]
    
    @classmethod
    def from_csharp(cls, params):
        thicks = _as_int16_view(params.Thicks, "Thicks", strict_zero_copy=True)
        return cls(thicks=thicks)
