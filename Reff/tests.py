from pathlib import Path
import shutil

import numpy as np
import os
import tempfile
from types import SimpleNamespace
from functools import wraps

from schemas import DataBatch, SearchParams, SupportData, TargetData, TargetType
from npz_exporter import save_to_npz
from run_plot import load_npz_data_batch

PATH_TO_TMP = Path(os.path.abspath(__file__)).parent / 'Tmp'


def print_status(_test_fn=None, *, progress_callback=None):
    def decorator(test_fn):
        @wraps(test_fn)
        def wrapper(*args, **kwargs):
            callback = kwargs.pop("progress_callback", None) or progress_callback
            if callback:
                callback(f"{test_fn.__name__}: start")
            try:
                result = test_fn(*args, **kwargs)
                print(f"[OK] {test_fn.__name__} successfully pass!")
                if callback:
                    callback(f"{test_fn.__name__}: success")
                return result
            except Exception:
                print(f"[FAIL] {test_fn.__name__}")
                if callback:
                    callback(f"{test_fn.__name__}: fail")
                raise
        return wrapper

    if _test_fn is None:
        return decorator
    return decorator(_test_fn)


@print_status(progress_callback=print)
def create_from_csharp():
    speed = np.array([10, 11, 12], dtype=np.int16)
    orientation = np.array([1, 2, 3], dtype=np.int16)
    axel_x = np.array([1, 2, 3], dtype=np.int16)
    axel_y = np.array([4, 5, 6], dtype=np.int16)
    axel_z = np.array([7, 8, 9], dtype=np.int16)
    raw_data = np.array([100, 200, 300, 400, 500, 600], dtype=np.int16)

    data_src = SimpleNamespace(
        Data=raw_data,
        ToolRotateAngle=orientation,
        Speed=speed,
        AxelX=axel_x,
        AxelY=axel_y,
        AxelZ=axel_z,
    )
    params_src = SimpleNamespace(StartDistance=5.5, OdomStep=0.25)
    target_points = np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.float64)
    target_labels = [TargetType.weld.value, TargetType.valve.value]
    target_src = SimpleNamespace(Points=target_points, Labels=target_labels)

    batch = DataBatch.from_csharp(data_src)
    params = SearchParams.from_csharp(params_src)
    target = TargetData.from_csharp(target_src)

    assert isinstance(batch, DataBatch)
    assert isinstance(params, SearchParams)
    assert batch.Accelerometer.shape == (3, 3)
    assert batch.Data.shape == (3, 2)
    assert np.array_equal(batch.Data, raw_data.reshape(3, 2))
    assert np.array_equal(batch.Orientation, orientation)
    assert np.array_equal(batch.Velocity, speed)
    assert batch.Accelerometer[0, 0] == data_src.AxelX[0]
    assert batch.Accelerometer[0, 1] == data_src.AxelY[0]
    assert batch.Accelerometer[0, 2] == data_src.AxelZ[0]
    assert np.shares_memory(batch.Velocity, speed)
    assert np.shares_memory(batch.Orientation, orientation)
    assert np.shares_memory(batch.Data, raw_data)
    assert params.StartDistance == params_src.StartDistance
    assert params.OdomStep == params_src.OdomStep
    assert isinstance(target, TargetData)
    assert target.points.shape == (2, 4)
    assert np.array_equal(target.points, target_points.reshape(2, 4))
    assert np.shares_memory(target.points, target_points)
    assert target.labels == [TargetType.weld, TargetType.valve]


@print_status(progress_callback=print)
def save_to_npz_case():
    speed = np.array([10, 11, 12], dtype=np.int16)
    orientation = np.array([1, 2, 3], dtype=np.int16)
    axel_x = np.array([1, 2, 3], dtype=np.int16)
    axel_y = np.array([4, 5, 6], dtype=np.int16)
    axel_z = np.array([7, 8, 9], dtype=np.int16)
    raw_data = np.array([100, 200, 300, 400, 500, 600], dtype=np.int16)
    thicks = np.array([9, 8, 7], dtype=np.int16)

    data_src = SimpleNamespace(
        Data=raw_data,
        ToolRotateAngle=orientation,
        Speed=speed,
        AxelX=axel_x,
        AxelY=axel_y,
        AxelZ=axel_z,
    )
    batch = DataBatch.from_csharp(data_src)

    params_src = SimpleNamespace(StartDistance=5.5, OdomStep=0.25)
    support_src = SimpleNamespace(thicks=thicks)
    target_points = np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.float64)
    target_labels = [TargetType.weld, TargetType.valve]
    target_src = SimpleNamespace(points=target_points, labels=target_labels)
    
    os.makedirs(PATH_TO_TMP, exist_ok=True)
    tmp_dir = PATH_TO_TMP
    save_to_npz(
        data=batch,
        searchParams=params_src,
        supportData=support_src,
        target_data=target_src,
        outputDirPath=tmp_dir,
        batch_idx=7,
    )

    batch_path = os.path.join(tmp_dir, "npz_datasets", "data_00007.npz")
    assert os.path.isfile(batch_path)

    with np.load(batch_path) as batch_npz:
        assert np.array_equal(batch_npz["magnetogram"], raw_data.reshape(3, 2))
        assert np.array_equal(batch_npz["velocity"], speed)
        assert np.array_equal(batch_npz["orientation"], orientation)
        assert np.array_equal(batch_npz["thicks"], thicks)
        assert np.array_equal(batch_npz["points"], target_points)
        assert np.array_equal(batch_npz["labels"], np.array(list(map(lambda x: x.value, target_labels)), dtype=np.int16))
        assert batch_npz["odomstep"] == params_src.OdomStep
    shutil.rmtree(PATH_TO_TMP)


@print_status(progress_callback=print)
def load_from_npz_case():
    speed = np.array([10, 11, 12], dtype=np.int16)
    orientation = np.array([1, 2, 3], dtype=np.int16)
    axel_x = np.array([1, 2, 3], dtype=np.int16)
    axel_y = np.array([4, 5, 6], dtype=np.int16)
    axel_z = np.array([7, 8, 9], dtype=np.int16)
    raw_data = np.array([100, 200, 300, 400, 500, 600], dtype=np.int16)
    thicks = np.array([9, 8, 7], dtype=np.int16)
    target_points = np.array([10, 11, 12, 13, 20, 21, 22, 23], dtype=np.float64)
    target_labels = [TargetType.weld, TargetType.valve]

    data_src = SimpleNamespace(
        Data=raw_data,
        ToolRotateAngle=orientation,
        Speed=speed,
        AxelX=axel_x,
        AxelY=axel_y,
        AxelZ=axel_z,
    )
    batch = DataBatch.from_csharp(data_src)
    params_src = SimpleNamespace(StartDistance=5.5, OdomStep=0.25)
    support_src = SimpleNamespace(thicks=thicks)
    target_src = SimpleNamespace(points=target_points, labels=target_labels)

    os.makedirs(PATH_TO_TMP, exist_ok=True)
    tmp_dir = PATH_TO_TMP
    save_to_npz(
        data=batch,
        searchParams=params_src,
        supportData=support_src,
        target_data=target_src,
        outputDirPath=tmp_dir,
        batch_idx=7,
    )
    datasets_dir = os.path.join(tmp_dir, "npz_datasets")
    batch, search_func = load_npz_data_batch(datasets_dir, 7)

    assert isinstance(batch, DataBatch)
    assert np.array_equal(batch.Data, raw_data.reshape(3, 2))
    assert np.array_equal(batch.Velocity, speed)
    assert np.array_equal(batch.Orientation, orientation)
    assert np.array_equal(batch.Accelerometer, np.column_stack((axel_x, axel_y, axel_z)))
    assert search_func.OdomStep == params_src.OdomStep
    shutil.rmtree(PATH_TO_TMP)

def unit_tests():
    """
    Need to check:
        - TODO: input params check
        - TODO: output format check
        - TODO: test cases
        - TODO: benchmark run (may be in another function)
    """
    create_from_csharp()
    save_to_npz_case()
    load_from_npz_case()

unit_tests()
