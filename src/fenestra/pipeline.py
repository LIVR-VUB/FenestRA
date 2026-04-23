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


def apply_post_processing(image: np.ndarray, clip_limit: float = 0.03, unsharp_radius: float = 1.0, unsharp_amount: float = 1.0) -> np.ndarray:
    """Applies robust mathematical contrast equalization and sharpening to highlight fenestration topology."""
    arr = image.astype(np.float64)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max > arr_min:
        arr_norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr_norm = np.zeros_like(arr)
        
    clahe = exposure.equalize_adapthist(arr_norm, clip_limit=clip_limit, nbins=256)
    sharp = filters.unsharp_mask(clahe, radius=unsharp_radius, amount=unsharp_amount)
    out_u16 = (sharp * np.iinfo(np.uint16).max).astype(np.uint16)
    return out_u16


def upsample_clahe(image: np.ndarray, factor: int = 4, clip_limit: float = 0.03, unsharp_radius: float = 1.0, unsharp_amount: float = 1.0):
    """Fallback Python CPU upsampling and CLAHE."""
    arr = image.astype(np.float64)
    arr_upsampled = zoom(arr, zoom=(factor, factor), order=3)
    return apply_post_processing(arr_upsampled, clip_limit, unsharp_radius, unsharp_amount)


@thread_worker
def run_dl_upsampling(
    temp_in_path: str,
    temp_out_dir: str,
    container_path: str,
    model_path: str,
    architecture: str,
    engine: str = "Singularity"
):
    """Runs the Singularly-contained DL upsampling in the background."""
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    
    if engine.strip().lower() == "singularity":
        cmd = [
            "singularity", "exec", "--nv",
            "--bind", f"{backend_dir}:/opt/dl_project/scripts",
            "--bind", f"{os.path.dirname(temp_in_path)}:/tmp_in",
            "--bind", f"{temp_out_dir}:/tmp_out",
            "--bind", f"{os.path.dirname(model_path)}:/tmp_model",
            container_path,
            "python", "/opt/dl_project/scripts/inference.py",
            "--input", "/tmp_in",
            "--output", "/tmp_out",
            "--model_path", f"/tmp_model/{os.path.basename(model_path)}",
            "--arch", architecture,
            "--tile_size", "256",
        ]
    elif engine.strip().lower() == "docker":
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{backend_dir}:/opt/dl_project/scripts",
            "-v", f"{os.path.dirname(temp_in_path)}:/tmp_in",
            "-v", f"{temp_out_dir}:/tmp_out",
            "-v", f"{os.path.dirname(model_path)}:/tmp_model",
            container_path,
            "python", "/opt/dl_project/scripts/inference.py",
            "--input", "/tmp_in",
            "--output", "/tmp_out",
            "--model_path", f"/tmp_model/{os.path.basename(model_path)}",
            "--arch", architecture,
            "--tile_size", "256",
        ]
    else:
        raise ValueError(f"Unknown engine: {engine}")
    
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
        bf16=False, # Disable BFloat16 for widespread legacy GPU compatibility
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


# =====================================================================
# Synchronous helpers for batch processing
# =====================================================================

def run_dl_upsampling_sync(
    temp_in_path: str,
    temp_out_dir: str,
    container_path: str,
    model_path: str,
    architecture: str,
    engine: str = "Singularity"
) -> str:
    """Synchronous DL upsampling — returns the output TIFF path directly."""
    import glob as _glob

    backend_dir = os.path.join(os.path.dirname(__file__), "backend")

    if engine.strip().lower() == "singularity":
        cmd = [
            "singularity", "exec", "--nv",
            "--bind", f"{backend_dir}:/opt/dl_project/scripts",
            "--bind", f"{os.path.dirname(temp_in_path)}:/tmp_in",
            "--bind", f"{temp_out_dir}:/tmp_out",
            "--bind", f"{os.path.dirname(model_path)}:/tmp_model",
            container_path,
            "python", "/opt/dl_project/scripts/inference.py",
            "--input", "/tmp_in",
            "--output", "/tmp_out",
            "--model_path", f"/tmp_model/{os.path.basename(model_path)}",
            "--arch", architecture,
            "--tile_size", "256",
        ]
    elif engine.strip().lower() == "docker":
        cmd = [
            "docker", "run", "--rm", "--gpus", "all",
            "-v", f"{backend_dir}:/opt/dl_project/scripts",
            "-v", f"{os.path.dirname(temp_in_path)}:/tmp_in",
            "-v", f"{temp_out_dir}:/tmp_out",
            "-v", f"{os.path.dirname(model_path)}:/tmp_model",
            container_path,
            "python", "/opt/dl_project/scripts/inference.py",
            "--input", "/tmp_in",
            "--output", "/tmp_out",
            "--model_path", f"/tmp_model/{os.path.basename(model_path)}",
            "--arch", architecture,
            "--tile_size", "256",
        ]
    else:
        raise ValueError(f"Unknown engine: {engine}")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Container DL Inference failed: {result.stderr}")

    out_files = _glob.glob(os.path.join(temp_out_dir, "*.tif*"))
    if not out_files:
        raise RuntimeError("No output generated from DL Upsampling.")
    return out_files[0]


def run_cellpose_sync(
    image: np.ndarray,
    model_path: str,
    diameter: float,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4
) -> np.ndarray:
    """Synchronous Cellpose segmentation — returns the masks array directly."""
    from cellpose import models
    import torch
    import re

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(model_path):
        model = models.CellposeModel(
            gpu=torch.cuda.is_available(),
            pretrained_model=model_path,
            device=device
        )
    else:
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

    return masks_arr[0] if isinstance(masks_arr, (tuple, list)) else masks_arr


# =====================================================================
# Batch pipeline orchestrator
# =====================================================================

@thread_worker
def run_batch_pipeline(
    input_dir: str,
    output_dir: str,
    method: str,
    # CLAHE params
    clahe_factor: int = 4,
    # Post-processing params
    clip_limit: float = 0.02,
    unsharp_radius: float = 1.0,
    unsharp_amount: float = 1.0,
    # DL params
    dl_model_path: str = "",
    container_path: str = "",
    engine: str = "Singularity",
    apply_postprocess: bool = False,
    # Cellpose params
    cp_model_path: str = "",
    diameter: float = 30.0,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
):
    """Process every .jpk-qi-image in input_dir and write consolidated Excel + TIFFs."""
    import glob as _glob

    os.makedirs(output_dir, exist_ok=True)

    # Discover all JPK files
    patterns = ["*.jpk-qi-image", "*.jpk"]
    jpk_files = []
    for pat in patterns:
        jpk_files.extend(_glob.glob(os.path.join(input_dir, pat)))
    jpk_files = sorted(set(jpk_files))

    if not jpk_files:
        raise RuntimeError(f"No .jpk-qi-image or .jpk files found in {input_dir}")

    total = len(jpk_files)
    all_dfs = []
    is_dl = "CLAHE" not in method
    architecture = "hat" if "HAT" in method else "swinir"

    for idx, jpk_path in enumerate(jpk_files, start=1):
        base_name = Path(jpk_path).stem
        yield f"Processing {idx}/{total}: {Path(jpk_path).name}"

        # --- Step 1: Load JPK ---
        raw_image, pixel_to_nm = process_jpk(jpk_path)

        # --- Step 2: Upsample ---
        if is_dl:
            # Create per-image temp dirs to avoid collisions
            with tempfile.TemporaryDirectory() as tmp_dir:
                temp_in_path = os.path.join(tmp_dir, "temp_in.tif")
                temp_out_dir_img = os.path.join(tmp_dir, "out")
                os.makedirs(temp_out_dir_img, exist_ok=True)

                tifffile.imwrite(temp_in_path, raw_image)
                out_tif = run_dl_upsampling_sync(
                    temp_in_path=temp_in_path,
                    temp_out_dir=temp_out_dir_img,
                    container_path=container_path,
                    model_path=dl_model_path,
                    architecture=architecture,
                    engine=engine
                )
                upsampled = tifffile.imread(out_tif)

            # Optional post-DL sharpening
            if apply_postprocess:
                upsampled = apply_post_processing(upsampled, clip_limit, unsharp_radius, unsharp_amount)

            upsample_factor = 4.0
        else:
            upsampled = upsample_clahe(raw_image, clahe_factor, clip_limit, unsharp_radius, unsharp_amount)
            upsample_factor = float(clahe_factor)

        # Save upsampled TIFF
        tifffile.imwrite(os.path.join(output_dir, f"{base_name}_upsampled.tif"), upsampled)

        # --- Step 3: Cellpose segmentation ---
        masks = run_cellpose_sync(
            image=upsampled,
            model_path=cp_model_path,
            diameter=diameter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
        )

        # Save mask TIFF
        tifffile.imwrite(os.path.join(output_dir, f"{base_name}_mask.tif"), masks)

        # --- Step 4: Quantify ---
        upsampled_scale = pixel_to_nm / upsample_factor
        df, porosity = quantify_fenestrations(masks, upsampled_scale, upsample_factor)
        df.insert(0, "Image_Name", base_name)
        df["Porosity"] = porosity
        all_dfs.append(df)

    # --- Step 5: Write consolidated Excel ---
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        excel_path = os.path.join(output_dir, "batch_results.xlsx")
        master_df.to_excel(excel_path, index=False, engine="openpyxl")

    yield f"BATCH_COMPLETE:{total}"
