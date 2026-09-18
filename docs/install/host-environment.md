# 1 - Host environment

The conda environment that runs napari, the FenestRA widget, Cellpose, and the JPK reader. This is
step 2 of **Option B — Native install (Linux)** in the README, split up here so you can see what
each line is for. On Windows or macOS, the [all-in-one container](all-in-one.md) replaces this page
entirely. Build it in the order below. The plugin itself is installed in
[2 - Install FenestRA](install-fenestra.md).

## Create the environment

```bash
conda create -n fenestra-env -c conda-forge python=3.10 numpy=1.26.4
conda activate fenestra-env
```

Python 3.10 is what the package declares support for (`python_requires = >=3.10,<3.13`) — 3.13 is
excluded deliberately, because pySPM 0.6.x declares `<3.13` and NumPy 1.26.4 has no wheels past
cp312, so pip on 3.13 fails at dependency resolution rather than telling you your Python is too
new.

NumPy is pinned to the 1.x series from the start so that conda does not resolve a NumPy 2 build that later pip
installs would have to fight with.

You need `conda activate fenestra-env` in every new terminal before launching napari.

## Install napari and the scientific stack

Every version here was read out of a working `fenestra-env` on 18 September 2026 — the same
environment this documentation's behavioural claims were verified against.

```bash
pip install "napari[all]==0.7.0" "PyQt6==6.11.0" "PyQt6-Qt6==6.11.0" "PyQt6-sip==13.11.1" \
            "qtpy==2.4.3" "magicgui==0.10.2" "superqt==0.8.1"

pip install "numpy==1.26.4" "scipy==1.15.3" "scikit-image==0.25.2" "pandas==2.3.3" \
            "tifffile==2025.5.10" "openpyxl==3.1.5"
```

| Package | Pin | Used for |
|---|---|---|
| `napari[all]` | `0.7.0` | the viewer and a Qt backend. The plain `napari` package ships no Qt bindings |
| `PyQt6` | `6.11.0` | `napari[all]` asks only for `PyQt6>6.5`, so two people installing a week apart get different Qt versions |
| `PyQt6-Qt6` | `6.11.0` | **the one that actually bites.** `PyQt6` pulls its Qt binaries as a separate package that floats within the minor series. A `PyQt6` whose version does not match its own `PyQt6-Qt6` is what produces `DLL load failed while importing QtWidgets` on Windows — see [Troubleshooting](../caveats/troubleshooting.md#no-qt-bindings-could-be-found) |
| `PyQt6-sip` | `13.11.1` | the binding layer between the two above |
| `qtpy`, `magicgui`, `superqt` | `2.4.3`, `0.10.2`, `0.8.1` | the widget layer the FenestRA dock is built on. `superqt` also owns the worker threading whose error handling FenestRA works around |
| `numpy` | `1.26.4` | not cosmetic: FenestRA declares `numpy>=1.26.0,<2.0.0`, and the surrounding stack is not NumPy 2 ready. NumPy 2 gives you import errors in dependencies rather than in FenestRA, which makes the cause hard to find |
| `scipy` | `1.15.3` | the cubic-spline zoom behind CLAHE upsampling |
| `scikit-image` | `0.25.2` | CLAHE, unsharp masking, `regionprops`, boundary finding for the overlay |
| `pandas` | `2.3.3` | the metrics table |
| `tifffile` | `2025.5.10` | reading and writing 16-bit and float32 AFM height TIFFs |
| `openpyxl` | `3.1.5` | writing `batch_results.xlsx` |

!!! note "These pins are not a `pip freeze` of that environment"

    The live `fenestra-env` carries pySPM 0.6.3 next to NumPy 1.26.4, a combination `pip check`
    reports as broken and that pip would refuse to reproduce from scratch. The set documented here
    is the resolvable equivalent, checked with `pip install --dry-run`; it lands on the same NumPy,
    SciPy, scikit-image and tifffile the working environment actually has.

## Install PyTorch against CUDA 12.4

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
            "torch==2.4.0" "torchvision==0.19.0"
```

!!! danger "Not for RTX 50-series / Blackwell (sm_120). Check your card first"

    ```bash
    nvidia-smi --query-gpu=name,compute_cap --format=csv
    ```

    The cu124 wheels pinned above carry kernels for `sm_50` … `sm_90` only (measured, not assumed:
    [known issue 13](../caveats/known-issues.md#13-rtx-50-series-blackwell-gpus-cannot-run-the-standard-image)).
    On a compute capability of 10.0 or 12.0 — an RTX 5060/5070/5080/5090, a B100 or a B200 — the
    card is still *detected*, `torch.cuda.is_available()` still returns `True`, and Cellpose is
    still constructed with `gpu=True`; the first actual kernel launch then raises
    `CUDA error: no kernel image is available for execution on the device`.

    Swapping the two lines above for `--index-url https://download.pytorch.org/whl/cu128` with
    `"torch==2.8.0" "torchvision==0.23.0"` fixes **host-side Cellpose only**. This page's DL
    backend is the reference container ([Container backend](container-backend.md)), which is torch
    1.14 with no PTX above `sm_86` and will not start on a Blackwell card at all — so on Blackwell
    the native install gives you CLAHE and Cellpose, and no DL upsampling. For DL upsampling on
    such a card the route is the `cu128` all-in-one image:
    [RTX 50-series and Blackwell: the cu128 image](all-in-one.md#rtx-50-series-and-blackwell-the-cu128-image).

    The compute-capability banner that catches this at startup exists only in the container
    launcher (`containers/entrypoint.sh`). The native path prints no such warning, which is why the
    check above is worth running before you install.

Install this **before** Cellpose. Cellpose declares PyTorch as a dependency, so if it goes first
pip satisfies that from PyPI with the default CPU wheel, and the CUDA build never gets installed.

The explicit index URL is what selects the CUDA 12.4 build. Without it you can end up with a
CPU-only PyTorch wheel, which installs cleanly and gives no error.

!!! warning "A CPU-only PyTorch is silent, not loud"

    FenestRA chooses its Cellpose device with `torch.cuda.is_available()`. If PyTorch cannot see
    your GPU, segmentation still runs, still produces masks, and still writes a CSV. It is only
    slower. Check the result of the install before you assume you are on the GPU:

    ```bash
    python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
    ```

    This is the host PyTorch. The container carries its own, older PyTorch and is unaffected by
    what you install here.

## Install Cellpose

```bash
pip install "cellpose==4.1.1"
```

Cellpose does the fenestration segmentation in panel 3. The documentation here describes
Cellpose 4 behavior, verified against 4.1.1. Two controls in panel 3 changed meaning between
Cellpose 2 and Cellpose 4 while their labels stayed the same, so read
[Segmentation](../guide/step3-segmentation.md) before trusting the **CP Model** and **Diameter**
fields.

## Install AFMReader

```bash
pip install "pySPM==0.6.2" "AFMReader==0.0.7"
```

AFMReader is the only reader for `.jpk-qi-image` files in the pipeline. It returns both the height
array and the scan's nanometers-per-pixel scale, and that scale is what converts every measurement
from pixels to nanometers.

`pySPM` is held below 0.6.3 because 0.6.3 declares `numpy>=2.0.0`, which would drag NumPy 2 into an
environment built around NumPy 1. It arrives as an AFMReader dependency; the `.jpk` code path never
imports it.

!!! note "This used to require Git, and no longer does"

    Earlier versions of these instructions installed AFMReader with
    `pip install git+https://github.com/AFM-SPM/AFMReader.git`, which shells out to `git` and
    stopped Windows users with `ERROR: Cannot find command 'git'`. AFMReader is published on PyPI,
    so plain `pip` is enough and Git is no longer a prerequisite.

## Check the environment before moving on

```bash
python -c "import napari, torch, cellpose, AFMReader; print('ok')"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import torch; print(torch.cuda.get_device_capability(), torch._C._cuda_getArchFlags())"
```

The first command should print `ok`. If it raises `ModuleNotFoundError`, the named package did not
install, and the plugin will fail later in a less obvious place. The second should report a
`+cu124` build and `True` on a machine with a working NVIDIA driver.

**`True` on its own is not enough.** It only says the driver and the card are visible, not that the
wheel contains kernels your card can run — on a Blackwell card `is_available()` is `True` and every
kernel launch still fails. That is what the third command checks: the capability it prints, written
as `sm_<major><minor>`, must appear in the arch flags beside it. `(8, 6)` against
`sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90` is fine; `(12, 0)` against the same list is the
Blackwell case above, and you should stop here rather than at **Run Upsampling** hours later.

Next: [2 - Install FenestRA](install-fenestra.md).
