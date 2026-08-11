# Getting Started

Installing FenestRA means building a conda environment, adding the plugin, and, if you want deep
learning upsampling, building a container and supplying model weights that are not public yet. This
page says what each step is for and which of the two platform paths applies to you.

Steps 1 to 3 follow the README's installation section directly. Steps 4 and 5 cover the two things
the README leaves to the Usage section.

## Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| Operating system | Linux, Windows, or macOS |
| GPU | NVIDIA card with CUDA 12.4 drivers recommended |
| Container engine | Apptainer on Linux, Docker Desktop on Windows and macOS |
| Environment manager | conda (the recipe uses `conda-forge`) |

A GPU is recommended rather than required. The **CLAHE (CPU)** upsampling method is pure CPU code,
and Cellpose falls back to the CPU when `torch.cuda.is_available()` is false. Deep learning
upsampling also falls back to the CPU inside the container, but a x4 transformer on CPU is slow
enough that it is not a practical route for a full scan.

## The steps

| Step | What it gives you | Needed for | README |
|---|---|---|---|
| [1 - Host environment](host-environment.md) | conda env, napari, Cellpose, AFMReader, PyTorch on CUDA 12.4 | everything | § 2 |
| [2 - Install FenestRA](install-fenestra.md) | the plugin itself, from PyPI | everything | § 3 |
| [3 - Container backend](container-backend.md) | the image that runs HAT and SwinIR | deep learning upsampling only | § 4 |
| [4 - Model weights](model-weights.md) | where the plugin looks for a `.pth` checkpoint (the trained weights are not public yet) | deep learning upsampling only | — |
| [5 - Verify the install](verify.md) | confidence that a scan goes in and a CSV comes out | everything | — |

If you only want the CLAHE path, steps 1, 2 and 5 are enough. Nothing in steps 3 and 4 is needed to
load a scan, segment it, and export measurements.

!!! danger "Maxwell-generation GPUs crash in Cellpose"

    GPUs with compute capability 5.2 or lower (Maxwell, for example the Quadro M4000) have no
    BFloat16 (`CUDA_R_16BF`) hardware. Cellpose 4 defaults to `use_bfloat16=True`, so on those
    cards PyTorch and Cellpose fail with `CUBLAS_STATUS_NOT_SUPPORTED`.

    FenestRA does not override that default. Keeping it preserves BFloat16 acceleration on
    current hardware, so the limitation is documented rather than worked around.

    The plugin has no device setting. It uses whatever `torch.cuda.is_available()` reports. To put
    Cellpose on the CPU on a Maxwell card, hide the GPU from the process that runs napari:

    ```bash
    CUDA_VISIBLE_DEVICES="" napari
    ```

## Which path am I on?

| | Linux | Windows / macOS |
|---|---|---|
| Container engine | Apptainer / Singularity | Docker Desktop |
| Build command | `sudo apptainer build dl_upsampling.sif containers/dl_upsampling.def` | `docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..` |
| **Engine** dropdown in panel 2 | `Singularity` | `Docker` |
| What you type next to it | the full path to your `.sif` file | the tag `livrvub/dl-upsampling:latest` |
| Status | working | the Docker argv is currently broken, see [Known issues](../caveats/known-issues.md) |

The CLAHE path behaves identically on all three operating systems, because it never touches a
container.

## Next

Start with [1 - Host environment](host-environment.md). If the environment is already built and
you want to know whether it works, jump to [5 - Verify the install](verify.md).
