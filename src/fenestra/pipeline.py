import functools
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


class FenestraError(Exception):
    """A worker failure that must reach the user.

    Deliberately NOT a RuntimeError. superqt's GeneratorWorker.work() catches RuntimeError,
    returns it instead of raising, and WorkerBase.run() then emits a warning and returns *before*
    `errored` and `finished` fire. The practical effect is the worst failure mode this codebase
    has: the QMessageBox never appears, the button stays greyed at "Upsampling in progress...",
    and the user has to restart napari to find out anything went wrong.

    Measured against the installed superqt: a @thread_worker generator raising RuntimeError
    produces no signals at all and leaves the worker reporting is_running == True.
    """


def _reporting(fn):
    """Re-raise RuntimeError from a worker generator as FenestraError so it reaches the GUI.

    This narrows nothing and swallows nothing — the original is chained with `from` and the
    error surfaces louder than before. It also rescues RuntimeErrors raised by libraries inside
    the worker, notably CUDA out-of-memory from torch or cellpose, which vanished the same way.
    """

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        try:
            yield from fn(*args, **kwargs)
        except RuntimeError as e:
            raise FenestraError(str(e)) from e

    return inner


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


# =====================================================================
# Deep-learning backend invocation
# =====================================================================
# One argv builder for every engine. The interactive worker and the batch loop both go through
# _run_dl_inference(), so the two can no longer drift apart the way the two hand-copied argv
# blocks used to.

#: Interpreter of the bundled deep-learning environment, used by the "Local" engine. Inside the
#: all-in-one image this is the second venv. Override with FENESTRA_DL_PYTHON anywhere else.
DEFAULT_DL_PYTHON = "/opt/venv-dl/bin/python"

DL_TILE_SIZE = 256


def _engine_key(engine: str) -> str:
    """Normalise a UI engine label ("Local (bundled)") down to its keyword ("local")."""
    parts = engine.strip().lower().split()
    return parts[0] if parts else ""


def _build_dl_cmd(engine, temp_in_path, temp_out_dir, container_path, model_path, architecture):
    """Return (argv, env) for one DL inference run. env is None to inherit the current one."""
    backend_dir = os.path.join(os.path.dirname(__file__), "backend")
    key = _engine_key(engine)

    if key == "local":
        python_exe = os.environ.get("FENESTRA_DL_PYTHON", DEFAULT_DL_PYTHON)
        if not os.path.exists(python_exe):
            raise RuntimeError(
                f"The bundled deep-learning environment was not found at {python_exe}. "
                "The Local engine exists only inside the FenestRA all-in-one container. "
                "Set FENESTRA_DL_PYTHON, or choose the Singularity or Docker engine."
            )
        # One filesystem, so there is nothing to bind-mount and no path to translate.
        cmd = [
            python_exe, os.path.join(backend_dir, "inference.py"),
            "--input", os.path.dirname(temp_in_path),
            "--output", temp_out_dir,
            "--model_path", model_path,
            "--arch", architecture,
            "--tile_size", str(DL_TILE_SIZE),
        ]
        # The GUI runs in a different venv. A PYTHONPATH or PYTHONHOME set for that one would drag
        # its numpy and torch into the DL interpreter, which is precisely what two separate
        # environments exist to prevent.
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        return cmd, env

    mounts = [
        (backend_dir, "/opt/dl_project/scripts"),
        (os.path.dirname(temp_in_path), "/tmp_in"),
        (temp_out_dir, "/tmp_out"),
        (os.path.dirname(model_path), "/tmp_model"),
    ]
    # Both engines need the explicit "python": singularity exec bypasses %runscript, and the
    # Docker recipe deliberately carries no ENTRYPOINT.
    inner = [
        "python", "/opt/dl_project/scripts/inference.py",
        "--input", "/tmp_in",
        "--output", "/tmp_out",
        "--model_path", f"/tmp_model/{os.path.basename(model_path)}",
        "--arch", architecture,
        "--tile_size", str(DL_TILE_SIZE),
    ]

    if key == "singularity":
        cmd = ["singularity", "exec", "--nv"]
        for host, guest in mounts:
            cmd += ["--bind", f"{host}:{guest}"]
    elif key == "docker":
        cmd = ["docker", "run", "--rm", "--gpus", "all"]
        for host, guest in mounts:
            cmd += ["-v", f"{host}:{guest}"]
    else:
        raise ValueError(f"Unknown engine: {engine}")

    return cmd + [container_path] + inner, None


def _run_dl_inference(temp_in_path, temp_out_dir, container_path, model_path, architecture, engine):
    """Run one DL upsampling and return the path of the TIFF it produced."""
    import glob as _glob

    cmd, env = _build_dl_cmd(
        engine, temp_in_path, temp_out_dir, container_path, model_path, architecture
    )

    # The interactive path reuses one output directory for the whole session. Without this, a run
    # that produces nothing leaves the previous run's TIFF in place, and the glob below happily
    # returns it — the user then segments and quantifies the wrong image with no indication.
    for stale in _glob.glob(os.path.join(temp_out_dir, "*.tif*")):
        os.remove(stale)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        raise RuntimeError(f"Container DL Inference failed: {result.stderr}")

    out_files = _glob.glob(os.path.join(temp_out_dir, "*.tif*"))
    if not out_files:
        raise RuntimeError("No output generated from DL Upsampling.")
    return out_files[0]


@thread_worker
@_reporting
def run_dl_upsampling(
    temp_in_path: str,
    temp_out_dir: str,
    container_path: str,
    model_path: str,
    architecture: str,
    engine: str = "Singularity"
):
    """Runs the DL upsampling in the background."""
    _run_dl_inference(
        temp_in_path=temp_in_path,
        temp_out_dir=temp_out_dir,
        container_path=container_path,
        model_path=model_path,
        architecture=architecture,
        engine=engine,
    )
    yield "done"


@thread_worker
@_reporting
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
        # No usable checkpoint: take cellpose's built-in default, which is cpsam. Cellpose >= 4.0.1
        # accepts model_type and ignores it, so the old "cyto2" argument never selected cyto2 —
        # it selected cpsam while telling the user otherwise.
        #
        # Announce it. Dropping that dead argument also dropped cellpose's own
        # "model_type argument is not used in v4.0.1+" line, which was the only signal a user ever
        # got that a different network had been substituted. Naming the path as well turns a
        # mistyped checkpoint — which is never validated before this point — from a silent
        # fallback into a visible one.
        print(
            f"FenestRA: no Cellpose checkpoint found at {model_path!r}; "
            "segmenting with the built-in default model (cpsam)."
        )
        model = models.CellposeModel(gpu=torch.cuda.is_available())

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
    return _run_dl_inference(
        temp_in_path=temp_in_path,
        temp_out_dir=temp_out_dir,
        container_path=container_path,
        model_path=model_path,
        architecture=architecture,
        engine=engine,
    )


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
        # See run_cellpose: model_type is accepted and ignored by cellpose >= 4.0.1, and this
        # print is the only thing that tells a batch user the default model was substituted.
        print(
            f"FenestRA: no Cellpose checkpoint found at {model_path!r}; "
            "segmenting with the built-in default model (cpsam)."
        )
        model = models.CellposeModel(gpu=torch.cuda.is_available())

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
@_reporting
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
