# How It Works

FenestRA runs its deep-learning step in a separate Python environment instead of inside the napari process. On the conda install that environment is a container; in the all-in-one images it is a second venv in the same image, reached by the **Local (bundled)** engine, which launches no container. This page explains why that split exists and what crosses the boundary between the two sides.

## Two environments that never share a process

The table below is the conda + separate-container install - the reference stack, built from `containers/dl_upsampling.def` or `containers/Dockerfile`, and the provenance for published numbers.

|  | Host | Reference-stack container |
|---|---|---|
| Base | conda `fenestra-env`, Python 3.10 | `nvcr.io/nvidia/pytorch:23.01-py3`, Python 3.8 / torch 1.14 |
| Holds | napari, Qt, Cellpose 4, AFMReader, torch 2.4 / cu124 | basicsr, HAT, SwinIR (git-cloned to `/opt`), opencv-headless 4.8.0.74, numpy<1.24 |
| Why | modern GUI stack | basicsr will not coexist with modern numpy and torch |

In the all-in-one images the same process split is two venvs - `/opt/venv-gui` and `/opt/venv-dl` - inside one image built on `nvidia/cuda:12.4.1-runtime-ubuntu22.04`. There is no conda environment and no nvcr base. See [Containers](containers.md).

The reason for the split is a dependency conflict, not a preference. HAT and SwinIR are built on `basicsr`, which relies on NumPy and PyTorch conventions that were removed in later releases. The container pins `numpy<1.24` for exactly that reason. Those pins cannot be satisfied at the same time as the versions napari and Cellpose 4 need, so FenestRA keeps two Pythons and never asks them to agree.

!!! info "The PyTorch 2.4 badge describes the host"
    The README badge refers to the environment napari and Cellpose run in. Inside the **reference** container (`containers/dl_upsampling.def`, `containers/Dockerfile`) the backend is torch 1.14, and it stays there - upgrading your host torch does not change what the super-resolution model runs on. The all-in-one images are different: their DL venv is torch 2.1.2 (`Dockerfile.allinone:129`) or torch 2.8.0/cu128 for Blackwell (`Dockerfile.allinone.cu128:160`). See [Containers](containers.md).

## The contact surface

Everything the two sides say to each other passes through one `subprocess.run` call, four bind mounts, and a temporary TIFF on disk:

1. The host writes the raw height array to `temp_in.tif` in a temporary directory.
2. The host builds an argv list (`_build_dl_cmd`, `pipeline.py:109-165`) and calls `subprocess.run` (`_run_dl_inference`, `pipeline.py:168-189`), on both the interactive and the batch path.
3. The container runs `inference.py`, reads the TIFF from `/tmp_in`, loads the weights from `/tmp_model`, and writes a float32 TIFF into `/tmp_out`.
4. The host reads that TIFF back and continues.

No Python object crosses the boundary. If the backend exits with a non-zero status, the host raises `RuntimeError: Container DL Inference failed: <stderr>` from `_run_dl_inference` (`pipeline.py:184`). Both the interactive and the batch path raise that same message, so the dialog shows the backend's standard error verbatim.

!!! note "This changed in 0.3.0"

    Before 0.3.0 the interactive path caught and re-wrapped the exception, so the dialog read `Background thread error: Container DL Inference failed: ...` while the batch path showed the unprefixed form. The extra wrapper is gone; if you are reading a `Background thread error:` prefix, you are on 0.2.11 or earlier.

    The message still says *Container* under the **Local (bundled)** engine, which launches no container. The wording is kept because [Troubleshooting](../caveats/troubleshooting.md) is indexed by that exact string.

The exact mount table and the three command shapes are on [Containers](containers.md).

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
- [Containers](containers.md) covers the image contents, the bind mounts, and the difference between the three engine commands - including the Local engine, which uses no container at all.
