import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import tifffile
from napari.qt.threading import thread_worker
from scipy.ndimage import zoom
from skimage import exposure, filters
import pandas as pd
from skimage.measure import regionprops, label

import sys
sys.path.append("/home/arka/Desktop/AFM-Project")
try:
    from AFMReader.jpk import load_jpk
except ImportError:
    print("Warning: AFMReader not found or could not be imported.")


def process_jpk(jpk_path: str, channel: str = "height_trace"):
    """Loads a JPK file and returns the image array and pixel-to-nm scale."""
    try:
        image, pixel_to_nm = load_jpk(
            file_path=jpk_path,
            channel=channel,
            flip_image=True,
        )
        return image, pixel_to_nm
    except Exception as e:
        raise RuntimeError(f"Failed to load JPK file: {e}")


def upsample_clahe(image: np.ndarray, factor: int = 4, clip_limit: float = 0.03, unsharp_radius: float = 1.0, unsharp_amount: float = 1.0):
    """Fallback Python CPU upsampling and CLAHE."""
    arr = image.astype(np.float64)
    arr_upsampled = zoom(arr, zoom=(factor, factor), order=3)
    
    arr_min, arr_max = arr_upsampled.min(), arr_upsampled.max()
    if arr_max > arr_min:
        arr_norm = (arr_upsampled - arr_min) / (arr_max - arr_min)
    else:
        arr_norm = np.zeros_like(arr_upsampled)
    
    clahe = exposure.equalize_adapthist(arr_norm, clip_limit=clip_limit, nbins=256)
    sharp = filters.unsharp_mask(clahe, radius=unsharp_radius, amount=unsharp_amount)
    out_u16 = (sharp * np.iinfo(np.uint16).max).astype(np.uint16)
    return out_u16


@thread_worker
def run_dl_upsampling(
    temp_in_path: str,
    temp_out_dir: str,
    container_path: str,
    model_path: str,
    architecture: str
):
    """Runs the Singularly-contained DL upsampling in the background."""
    inference_script = "/home/arka/Desktop/AFM-Project/DL_Upsampling/scripts/inference.py"
    
    cmd = [
        "singularity", "exec", "--nv",
        "--bind", f"{os.path.dirname(temp_in_path)}:/tmp_in",
        "--bind", f"{temp_out_dir}:/tmp_out",
        "--bind", f"{os.path.dirname(model_path)}:/tmp_model",
        "--bind", "/home/arka/Desktop/AFM-Project/DL_Upsampling:/opt/dl_project",
        container_path,
        "python", "/opt/dl_project/scripts/inference.py",
        "--input", "/tmp_in",
        "--output", "/tmp_out",
        "--model_path", f"/tmp_model/{os.path.basename(model_path)}",
        "--arch", architecture,
        "--tile_size", "256",  # safe tiling for big afm images
    ]
    
    try:
        # We invoke this as a subprocess
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"Container DL Inference failed: {result.stderr}")
            
        yield "done"
    except Exception as e:
        raise RuntimeError(f"Background thread error: {e}")


@thread_worker
def run_cellpose(image: np.ndarray, model_path: str, diameter: float, cellprob_threshold: float = 0.0, flow_threshold: float = 0.4):
    """Runs cellpose segmentation in the background."""
    from cellpose import models
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize Cellpose model
    if os.path.exists(model_path):
        model = models.CellposeModel(
            gpu=torch.cuda.is_available(),
            pretrained_model=model_path,
            device=device
        )
    else:
        # Fallback to cyto2 if default empty
        model = models.CellposeModel(gpu=torch.cuda.is_available(), model_type="cyto2")

    eval_kwargs = dict(
        channels=None,
        channel_axis=None,
        normalize={"normalize": True, "percentile": (1.0, 99.0)},
        diameter=diameter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        min_size=15,
        do_3D=False,
        augment=False,
        tile=True,
    )
    
    import re
    bad_keys = set()
    attempt = 0
    cur = dict(eval_kwargs)
    masks_arr = None
    
    while True:
        attempt += 1
        try:
            masks_arr = model.eval(image, **cur)
            break
        except TypeError as e:
            msg = str(e)
            m = re.search(r"unexpected keyword argument '([^']+)'", msg)
            if not m:
                raise
            bad = m.group(1)
            if bad in cur:
                bad_keys.add(bad)
                cur.pop(bad, None)
                if attempt > 12:
                    raise RuntimeError(f"Too many eval() retries; unsupported keys: {sorted(bad_keys)}")
            else:
                raise
                
    yield masks_arr[0] if isinstance(masks_arr, (tuple, list)) else masks_arr


def quantify_fenestrations(masks: np.ndarray, upsampled_scale_nm: float, upsample_factor: float):
    """Extract fenestration size, perimeter, and porosity.
    Upsampled scale nm is how many nm 1 pixel represents.
    """
    props = regionprops(masks)
    
    data = []
    # Convert pixels to physical nm properties
    pixel_area = upsampled_scale_nm ** 2
    pixel_length = upsampled_scale_nm
    
    for p in props:
        data.append({
            "Label": p.label,
            "Area_nm2": p.area * pixel_area,
            "Perimeter_nm": p.perimeter * pixel_length,
            "Equivalent_Diameter_nm": p.equivalent_diameter * pixel_length,
            "Equivalent_Diameter_Upsampled_Pixels": p.equivalent_diameter,
            "Equivalent_Diameter_Raw_Pixels": p.equivalent_diameter / upsample_factor,
            "Eccentricity": p.eccentricity
        })
        
    df = pd.DataFrame(data)
    
    # Porosity is the total mask area / total image area
    total_area_nm2 = masks.size * pixel_area
    total_fenestrations_area_nm2 = df["Area_nm2"].sum() if not df.empty else 0
    porosity = total_fenestrations_area_nm2 / total_area_nm2
    
    return df, porosity
