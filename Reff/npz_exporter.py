import os
import sys
from pathlib import Path
import numpy as np


def get_utils_dir():
    file_path = os.path.abspath(__file__)
    p = Path(file_path)
    dir_path = str(p.parent.parent
                   )
    return dir_path
sys.path.append(get_utils_dir())

from utils import convert_csharp_models
from schemas import DataBatch, SearchParams, SupportData, TargetData

DATASETS_FOLDER_NAME = "npz_datasets"


def create_npz_paths(outputDirPath):
    """
    Try to create tuple of npz paths in output directory, return empty list if error occurs
    """
    if not os.path.isdir(outputDirPath):
        raise NotADirectoryError(f"Output directory '{outputDirPath}' does not exist.")
    try:
        datasets_folder = os.path.join(outputDirPath, DATASETS_FOLDER_NAME)
        os.makedirs(datasets_folder, exist_ok=True)
        return datasets_folder
    except Exception as e:
        raise RuntimeError(f"Error accessing output directory: {e}") from e

def save_data_batch_to_npz(data_batch: DataBatch, 
                           outputDirPath: str, 
                           batch_idx: int,
                           support_data: SupportData,
                           target_data: TargetData,
                           search_params: SearchParams):
    """
    Save data batch to npz file in output directory with name 
    "output_{batch_idx:05d}.npz"
    """
    np.savez(
        os.path.join(outputDirPath, f"data_{batch_idx:05d}.npz"),
        magnetogram=data_batch.Data,
        velocity=data_batch.Velocity,
        accelerometer=data_batch.Accelerometer,
        orientation=data_batch.Orientation,
        thicks=support_data.thicks if support_data else np.array([], dtype=np.int16),
        points=target_data.points,
        labels=np.array([label.value for label in target_data.labels], dtype=np.int16),
        odomstep=search_params.OdomStep
    )


# @convert_csharp_models
def save_to_npz(data: DataBatch, 
                searchParams: SearchParams,
                supportData: SupportData,
                target_data: TargetData,
                outputDirPath: str,
                batch_idx: int):
    
    data_batches_folder = create_npz_paths(outputDirPath)
    save_data_batch_to_npz(
        data,
        data_batches_folder,
        batch_idx,
        supportData,
        target_data=target_data,
        search_params=searchParams
    )