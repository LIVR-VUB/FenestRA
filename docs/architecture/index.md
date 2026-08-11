# How It Works

FenestRA runs its deep-learning step inside a container instead of inside the napari process. This page explains why that split exists and what crosses the boundary between the two sides.

## Two environments that never share a process

|  | Host | Container |
|---|---|---|
| Base | conda `fenestra-env`, Python 3.10 | `nvcr.io/nvidia/pytorch:23.01-py3`, Python 3.8 / torch 1.14 |
| Holds | napari, Qt, Cellpose 4, AFMReader, torch 2.4 / cu124 | basicsr, HAT, SwinIR (git-cloned to `/opt`), opencv-headless 4.8.0.74, numpy<1.24 |
| Why | modern GUI stack | basicsr will not coexist with modern numpy and torch |

The reason for the split is a dependency conflict, not a preference. HAT and SwinIR are built on `basicsr`, which relies on NumPy and PyTorch conventions that were removed in later releases. The container pins `numpy<1.24` for exactly that reason. Those pins cannot be satisfied at the same time as the versions napari and Cellpose 4 need, so FenestRA keeps two Pythons and never asks them to agree.

!!! info "The PyTorch 2.4 badge describes the host"
    The README badge refers to the environment napari and Cellpose run in. Inside the container the version is torch 1.14, and it stays there. Upgrading your host torch does not change what the super-resolution model runs on.

## The contact surface

Everything the two sides say to each other passes through one `subprocess.run` call, four bind mounts, and a temporary TIFF on disk:

1. The host writes the raw height array to `temp_in.tif` in a temporary directory.
2. The host builds an argv list and calls `subprocess.run` (`pipeline.py:66-99` for the interactive path, `pipeline.py:224-257` for the batch path).
3. The container runs `inference.py`, reads the TIFF from `/tmp_in`, loads the weights from `/tmp_model`, and writes a float32 TIFF into `/tmp_out`.
4. The host reads that TIFF back and continues.

No Python object crosses the boundary. If the container exits with a non-zero status, the host raises `RuntimeError: Container DL Inference failed: <stderr>`. On the interactive path that exception is caught and re-wrapped (`pipeline.py:113-114`), so the message box reads `Background thread error: Container DL Inference failed: <stderr>`. The batch path raises the unprefixed form (`pipeline.py:261`). Either way the container's standard error is shown verbatim.

The exact mount table and the two command shapes are on [Containers](containers.md).

## Nothing to check out

`inference.py` ships inside the pip package. The host resolves the script directory as `os.path.dirname(__file__)/backend`, which points into your site-packages, and bind-mounts it to `/opt/dl_project/scripts`. You do not need the `DL_Upsampling` research repository on disk to run the plugin. You do need the container image, and you need the model weights, which are a separate file you supply. See [Model Weights](../install/model-weights.md).

## What runs where

| Step | Runs on |
|---|---|
| Load `.jpk-qi-image` (AFMReader) | Host |
| CLAHE upsampling | Host, CPU |
| HAT / SwinIR upsampling | Container, GPU |
| Optional post-DL sharpening | Host, CPU |
| Cellpose segmentation | Host GPU |
| `regionprops`, metrics, CSV and XLSX export | Host |

!!! note
    The container recipes install `cellpose>=2.2`, but FenestRA never calls it there. Segmentation always runs on the host, against whichever Cellpose version your conda environment has. That is why the Cellpose 4 behavior changes described in [Known Issues](../caveats/known-issues.md) apply regardless of which container you built.

## Next

- [Dataflow](dataflow.md) walks the array through every stage, with dtypes and value ranges.
- [Containers](containers.md) covers the image contents, the bind mounts, and the difference between the two engine commands.
