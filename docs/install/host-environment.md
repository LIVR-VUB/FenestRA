# 1 - Host environment

The conda environment that runs napari, the FenestRA widget, Cellpose, and the JPK reader. This is
step 2 of the README's installation section, split up here so you can see what each line is for.
Build it in the order below. The plugin itself is installed in [step 2](install-fenestra.md).

## Create the environment

```bash
conda create -n fenestra-env -c conda-forge python=3.10 numpy=1.26.4
conda activate fenestra-env
```

Python 3.10 is what the package declares support for (`python_requires = >=3.10`). NumPy is pinned
to the 1.x series from the start so that conda does not resolve a NumPy 2 build that later pip
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
```

The first command should print `ok`. If it raises `ModuleNotFoundError`, the named package did not
install, and the plugin will fail later in a less obvious place. The second should report a
`+cu124` build and `True` on a machine with a working NVIDIA driver.

Next: [2 - Install FenestRA](install-fenestra.md).
