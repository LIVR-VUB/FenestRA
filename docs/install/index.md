# Getting Started

Installing FenestRA means building a conda environment, adding the plugin, and, if you want deep
learning upsampling, building a container and supplying model weights that are not public yet. This
page says what each step is for and which of the two platform paths applies to you.

Steps 1 to 3 follow README **Option B — Native install** directly. Steps 4 and 5 cover the two things
the README leaves to the Usage section.

!!! tip "On Windows or macOS, you probably want the container instead"

    [The all-in-one container](all-in-one.md) replaces every step on this page with one
    `docker build` and one double-click. Docker Desktop becomes the only thing installed on your
    machine: no conda, no Qt binding, no CUDA PyTorch, no Git, and no separate backend image. It
    exists specifically because those steps keep failing on Windows.

    The page below remains the right route on Linux, **except on an RTX 50-series (Blackwell)
    card**, which neither the host PyTorch of step 1 nor the reference container of step 3 can
    drive — see the Blackwell note below. On every other card it is the route to use for numbers
    you intend to publish — see the stack caveat on that page. A Blackwell card cannot produce
    reference-stack numbers at all.

## Requirements

| | |
|---|---|
| Python | 3.10, 3.11 or 3.12 (`python_requires = >=3.10,<3.13`). 3.13 is not supported: pySPM 0.6.x and numpy 1.26.4 have no cp313 wheels, so `pip install napari-fenestra` fails to resolve. |
| Operating system | Linux, Windows, or macOS |
| GPU | NVIDIA card with CUDA 12.4 drivers recommended. **RTX 50-series (Blackwell, compute capability 12.0 / `sm_120`) cannot use this path** — see below. |
| Container engine | Apptainer on Linux, Docker Desktop on Windows and macOS |
| Environment manager | conda (the recipe uses `conda-forge`) |

A GPU is recommended rather than required. The **CLAHE (CPU)** upsampling method is pure CPU code,
and Cellpose falls back to the CPU when `torch.cuda.is_available()` is false. Deep learning
upsampling also falls back to the CPU inside the container, but a x4 transformer on CPU is slow
enough that it is not a practical route for a full scan.

## The steps

| Step | What it gives you | Needed for | README |
|---|---|---|---|
| [1 - Host environment](host-environment.md) | conda env, napari, Cellpose, AFMReader, PyTorch on CUDA 12.4 | everything | Option B § 2 |
| [2 - Install FenestRA](install-fenestra.md) | the plugin itself, from PyPI | everything | Option B § 3 |
| [3 - Container backend](container-backend.md) | the image that runs HAT and SwinIR | deep learning upsampling only | Option B § 4 |
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

!!! danger "RTX 50-series (Blackwell) GPUs cannot run this page's stack at all"

    Blackwell cards (RTX 5060 to 5090, compute capability 12.0, `sm_120`) have no kernels in
    either half of this path. The host install of step 1 pins `torch 2.4.0+cu124`, whose
    architecture list stops at `sm_90`; the reference container of step 3 is `torch 1.14`, which
    stops at `sm_86`.

    The only working route is the `cu128` all-in-one image,
    `containers/Dockerfile.allinone.cu128`, selected with `FENESTRA_IMAGE=livrvub/fenestra:cu128`
    — see
    [RTX 50-series and Blackwell: the cu128 image](all-in-one.md#rtx-50-series-and-blackwell-the-cu128-image).

    It also means a Blackwell card cannot produce reference-stack numbers, because the reference
    container will not start on it. See
    [Known issues](../caveats/known-issues.md#13-rtx-50-series-blackwell-gpus-cannot-run-the-standard-image).

## Which path am I on?

| | Linux | Windows / macOS |
|---|---|---|
| Container engine | Apptainer / Singularity | Docker Desktop |
| Build command | from the repository root: `sudo apptainer build dl_upsampling.sif containers/dl_upsampling.def` | from the repository root: `cd containers && docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..` (the trailing `..` is the build context, so the command only resolves from inside `containers/`) |
| **Engine** dropdown in panel 2 | `Singularity` | `Docker` |
| What you type next to it | the full path to your `.sif` file | the tag `livrvub/dl-upsampling:latest` |
| Status | working | working; rebuild any image built from a checkout older than the ENTRYPOINT fix (18 September 2026), see [Known issues](../caveats/known-issues.md) |

The CLAHE path behaves identically on all three operating systems, because it never touches a
container.

## Next

Start with [1 - Host environment](host-environment.md). If the environment is already built and
you want to know whether it works, jump to [5 - Verify the install](verify.md).
