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

```bash
pip install "napari[all]" "PyQt6==6.11.0" magicgui qtpy scipy scikit-image pandas tifffile "numpy<2" openpyxl
```

| Package | Used for |
|---|---|
| `napari[all]` | the viewer and a Qt backend. The plain `napari` package ships no Qt bindings |
| `PyQt6==6.11.0` | pinned deliberately. `napari[all]` resolves to `PyQt6>6.5` with no upper bound, so two people installing a week apart get different Qt versions — and a `PyQt6` whose version does not match its own `PyQt6-Qt6` produces `DLL load failed while importing QtWidgets` on Windows. See [Troubleshooting](../caveats/troubleshooting.md#no-qt-bindings-could-be-found) |
| `magicgui`, `qtpy` | the widget layer the FenestRA dock is built on |
| `scipy` | the cubic-spline zoom behind CLAHE upsampling |
| `scikit-image` | CLAHE, unsharp masking, `regionprops`, boundary finding for the overlay |
| `pandas` | the metrics table |
| `tifffile` | reading and writing 16-bit and float32 AFM height TIFFs |
| `openpyxl` | writing `batch_results.xlsx` |
| `numpy<2` | the pin, repeated for pip |

The NumPy pin is not cosmetic. FenestRA declares `numpy>=1.26.0,<2.0.0`, and the surrounding stack
it runs against is not NumPy 2 ready. Installing NumPy 2 into this environment gives you import
errors in dependencies rather than in FenestRA itself, which makes the cause hard to see.

## Install PyTorch against CUDA 12.4

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.0 torchvision==0.19.0
```

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
pip install cellpose==4.1.1
```

Cellpose does the fenestration segmentation in panel 3. The documentation here describes
Cellpose 4 behavior, verified against 4.1.1. Two controls in panel 3 changed meaning between
Cellpose 2 and Cellpose 4 while their labels stayed the same, so read
[Segmentation](../guide/step3-segmentation.md) before trusting the **CP Model** and **Diameter**
fields.

## Install AFMReader

```bash
pip install "pySPM<0.6.3" "AFMReader==0.0.7"
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
