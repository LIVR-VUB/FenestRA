<p align="center">
  <img src="https://raw.githubusercontent.com/LIVR-VUB/FenestRA/main/misc/FenestRA.jpg" alt="FenestRA Logo" width="450"/>
</p>

# FenestRA
**Fenestration Resolution & Analysis Pipeline**

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Napari](https://img.shields.io/badge/napari-plugin-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg?logo=pytorch)
[![DOI](https://zenodo.org/badge/1213499953.svg)](https://doi.org/10.5281/zenodo.19700659)
[![PyPI](https://img.shields.io/pypi/v/napari-fenestra.svg?labelColor=000000&color=blue)](https://pypi.org/project/napari-fenestra/)


FenestRA is a custom Napari plugin built for the Advanced LSEC AFM Pipeline. It bridges the gap between interactive Napari features, legacy deep-learning upscale repositories via containerized backends, and state-of-the-art Cellpose instance segmentation. 

By combining Deep Learning-based Super Resolution (HAT / SwinIR) with automated morphological analysis, FenestRA drastically simplifies the workflow of extracting robust physical porosity and fenestration morphology metrics directly from raw `.jpk.qi-image` files.

> [!IMPORTANT]
> **Pre-Publication Notice**  
> This repository provides the public codebase and scaffolding for the FenestRA pipeline. The fine-tuned deep learning model weights (specifically for the HAT, SwinIR, and custom Cellpose LSEC segmentation models) are currently kept private. They will be made completely publicly available alongside the peer-reviewed manuscript immediately upon its formal publication.

---

## Features

- **Cross-Platform Container Engine:** Seamlessly toggle between **Docker** (Windows / macOS) and **Singularity / Apptainer** (Linux / HPC) directly from the Napari UI. No code changes needed when switching platforms.
- **Hub-and-Spoke Deep Learning Architecture:** Run legacy Python 3.8 dependent upscale models (HAT, SwinIR) asynchronously inside a container without freezing your modern Napari GUI.
- **Post-DL Image Enhancement:** Optional CLAHE contrast equalization and Unsharp Masking applied directly to the Deep Learning output to sharpen fenestration edges before segmentation.
- **Native JPK Ingestion:** Automatically reads native physical scale (`nm / px`) from `.jpk.qi-image` files using AFMReader.
- **Synchronized 4-Pane Analysis:** Auto-generates a synchronized Napari viewer layout combining Raw, Upsampled, Mask, and Boundary Overlays natively.
- **Configurable CPU Fallback:** Includes high-fidelity Python-based CLAHE and unsharp masking functions when DL inference isn't required.
- **Sub-cellular Quantification:** Automatically calculates standard metrics (area, perimeter, equivalent diameter, eccentricity, porosity) with digital-to-physical size translations directly to `.csv`.
- **Batch Analysis:** Process an entire folder of `.jpk-qi-image` files in one automated run. Produces a single consolidated `.xlsx` Excel file with metrics from all images, plus individual upsampled TIFFs and Cellpose mask TIFFs.

---

## Installation

> [!TIP]
> These are the condensed steps. The [documentation site](docs/install/index.md) walks through the
> same install with per-package explanations, a verification checklist, and a troubleshooting page
> indexed by error message. Build it locally with `bash website/serve.sh` (see
> [Documentation](#documentation)).

FenestRA installs two ways. Pick one.

| | Option A — All-in-one container | Option B — Native install |
|---|---|---|
| Best for | **Windows and macOS** | **Linux** |
| You install | Docker Desktop, nothing else | Anaconda, Qt, CUDA PyTorch, Cellpose, Apptainer |
| Interface | napari in your web browser | napari as a desktop app |
| Use it for publication numbers | no — see the note in step 7 | yes |

---

## Option A — All-in-one container (Windows & macOS)

One image holds napari, FenestRA, Cellpose and the super-resolution backend. Nothing Python-shaped
is installed on your machine, which removes every Windows failure we have actually been sent:
`QtBindingsNotFoundError`, `DLL load failed while importing QtWidgets`,
`DLL load failed ... application control policies have blocked this file`, and
`ERROR: Cannot find command 'git'`.

Full guide, with troubleshooting: **[All-in-one container](docs/install/all-in-one.md)**.

### 1. Install Docker Desktop

Download it from [docker.com](https://www.docker.com/products/docker-desktop/) and install it.

On **Windows**, during setup choose the **WSL 2 backend** (the default). Start Docker Desktop and
wait until the whale icon in the system tray stops animating.

### 2. For the deep-learning methods, check your GPU

Install a current **NVIDIA driver**. Docker Desktop on Windows exposes the GPU through WSL 2, and
the launcher passes `--gpus all` for you.

> [!NOTE]
> Without a working NVIDIA GPU the app still starts. CLAHE (CPU) upsampling works, Cellpose falls
> back to the CPU and becomes slow, and HAT/SwinIR cannot run. The launcher says so at startup
> rather than letting you find out from the clock. **macOS has no GPU path at all** — Docker
> Desktop has no NVIDIA passthrough.

### 3. Get the repository

```bash
git clone https://github.com/LIVR-VUB/FenestRA.git
cd FenestRA
```

Git is needed only for this step, to fetch the recipe. It is not a dependency of the software.

### 4. Build the image, once

**Windows** (Command Prompt, from the `FenestRA` folder):

```bat
docker build -t livrvub/fenestra:latest -f containers\Dockerfile.allinone .
```

**Linux / macOS:**

```bash
docker build -t livrvub/fenestra:latest -f containers/Dockerfile.allinone .
```

This downloads several gigabytes and takes a while. You do it once. Budget about **17 GB of disk**
for the finished image.

The build fails rather than completes if the Qt plugin cannot load, if `basicsr` cannot be
imported, if the pinned HAT checkout is wrong, or if the Cellpose weights arrive truncated — so a
build that succeeds is worth something.

> [!WARNING]
> **Linux users whose Docker came from snap:** the snap cannot read `/media` or `/mnt` at all, so
> building from a repository on an external drive fails with `transferring dockerfile: 2B` and
> `no such file or directory` even though the file is plainly there. Run
> `sudo snap connect docker:removable-media`, or clone under your home directory.

### 5. Put your files where the app can see them

The launcher creates these two folders on first run and mounts them into the app:

| On your machine | Inside FenestRA | Put here |
|---|---|---|
| `%USERPROFILE%\FenestRA\data` (Windows)<br>`~/FenestRA/data` (Linux/macOS) | `/data` | your `.jpk-qi-image` scans; your results |
| `%USERPROFILE%\FenestRA\models`<br>`~/FenestRA/models` | `/models` | your `.pth` checkpoints |

Anything saved **outside** those two folders lives inside the container and is lost when you close
it. Save your CSV and batch output under `/data`.

> [!IMPORTANT]
> **Inside FenestRA you will only see these two folders.** The container cannot reach the rest of
> your PC — that is the isolation working, not a fault. So if napari's file dialog shows nothing but
> `/data`, your scans are not in the mounted folder yet.
>
> Either copy them into `%USERPROFILE%\FenestRA\data`, or point the launcher at wherever they
> already live, without moving anything:
>
> ```bat
> cd FenestRA
> set FENESTRA_DATA=D:\Microscopy\LSEC
> containers\run_fenestra.bat
> ```
>
> On Linux and macOS the same variable works: `FENESTRA_DATA=/path/to/scans containers/run_fenestra.sh`.
> `FENESTRA_MODELS` does the same for checkpoints. Launch from that same Command Prompt rather than
> double-clicking, or the variable will not be set.

The trained weights are not distributed until the manuscript is published — see
[Model weights](docs/install/model-weights.md).

### 6. Start it

**Windows:** double-click `containers\run_fenestra.bat`.

**Linux / macOS:**

```bash
containers/run_fenestra.sh
```

Leave that window open — it is the app's log, and it prints whether a GPU was found and whether
`/models` is empty. Closing it, or pressing `Ctrl+C`, shuts FenestRA down.

### 7. Open it in your browser

Go to **<http://localhost:6080>**. napari appears with the FenestRA dock already open, and panel 2
is preset to the bundled backend — the **Engine** dropdown reads `Local (bundled)` and **DL Model**
is pre-filled with `/models/best_model_ema.pth`.

![FenestRA running in a browser](docs/assets/ui/all-in-one-desktop.png)

Usage from here is identical to the desktop app — see [Usage](#usage).

> [!CAUTION]
> The launchers publish the desktop to `127.0.0.1` only, meaning *this machine*. If you change
> that to `-p 6080:6080`, Docker binds **every** network interface, and anyone on the same office
> LAN, conference wifi or hotel network can open `http://your-machine:6080` and get a fully
> interactive session — including a file dialog onto every folder you mounted. VNC carries no
> encryption, so a password does not fix it. There is no error and no warning; it simply works,
> for them. Set `VNC_PASSWORD` for a second layer if you like, but the loopback publish is the
> real control.

> [!IMPORTANT]
> **Publication numbers should not come from this image.** Its bundled backend runs torch 2.1.2
> rather than the reference stack's torch 1.14, because torch 1.14 will not start on any GPU newer
> than sm_86 — no RTX 40-series, no H100. The architectures and weights are identical, but the two
> stacks are not bit-for-bit equivalent. Build the reference container
> (`containers/dl_upsampling.def`, Option B step 4) for anything destined for a manuscript.

### Screen size

**You normally do not need to set anything.** The desktop resizes itself to your browser window, so
maximising the browser — or pressing `F11` for full screen — already gives you your monitor's full
resolution, with nothing scaled or blurred. Resize the browser and the desktop follows.

`SCREEN` sets only the size the desktop starts at, before a browser has connected. Set it if that
first moment matters, or if you are connecting with a VNC client that cannot resize:

**Windows** — set it in the same Command Prompt, then start FenestRA from there rather than
double-clicking:

```bat
cd FenestRA
set SCREEN=2560x1440
containers\run_fenestra.bat
```

**Linux / macOS:**

```bash
SCREEN=2560x1440 containers/run_fenestra.sh
```

Both `2560x1440` and the older `2560x1440x24` form are accepted. `VNC_PASSWORD` is set the same
way, if you want a password on the desktop as well as the loopback-only port:

```bat
set VNC_PASSWORD=something-long
containers\run_fenestra.bat
```

### Updating

```bash
git pull
docker build -t livrvub/fenestra:latest -f containers/Dockerfile.allinone .
```

Unchanged layers are reused, so an update is much faster than the first build.

---

## Option B — Native install (Linux)

The right choice on Linux, where Apptainer works properly and there is no VNC layer between you and
the GPU. This is also the route that builds the reference stack.

### 1. Requirements
- Python 3.10+
- An NVIDIA GPU with CUDA 12.4 drivers (recommended for DL inference)
- ~~**Git.**~~ No longer required: `AFMReader` is on PyPI and installs with plain `pip`.

> [!CAUTION]
> <small>**Hardware Compatibility Warning:** FenestRA requires deep learning hardware capable of running modern tensor operations. Extremely old legacy GPUs based on the Maxwell architecture (Compute Capability 5.2 or earlier, such as the Quadro M4000) physically lack hardware support for BFloat16 (`CUDA_R_16BF`) math. Running the plugin on these ancient GPUs will cause PyTorch and Cellpose to instantly crash with a `CUBLAS_STATUS_NOT_SUPPORTED` error.</small>

- **Linux:** Apptainer / Singularity
- **Windows / macOS:** Docker Desktop

### 2. Create the Host Environment
Create a clean Anaconda environment optimized for Cellpose targeting CUDA 12.4:

```bash
conda create -n fenestra-env -c conda-forge python=3.10 numpy=1.26.4
conda activate fenestra-env

# Install base GUI tools, Napari, and core scientific dependencies
# PyQt6 is pinned: napari[all] resolves to an unbounded PyQt6>6.5, and a PyQt6 whose version
# does not match its own PyQt6-Qt6 is what produces "DLL load failed while importing QtWidgets"
pip install "napari[all]" "PyQt6==6.11.0" magicgui qtpy scipy scikit-image pandas tifffile "numpy<2" openpyxl

# Install PyTorch mapped explicitly to CUDA 12.4 to ensure GPU hardware acceleration works
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.0 torchvision==0.19.0

# Install Cellpose for fenestration instance segmentation (pinned: the docs and UI labels describe 4.1.1 behavior)
pip install cellpose==4.1.1

# Install AFMReader for handling raw JPK AFM metadata.
# pySPM is held below 0.6.3 because 0.6.3 requires NumPy 2; the .jpk path never imports it.
pip install "pySPM<0.6.3" "AFMReader==0.0.7"
```

### 3. Install FenestRA
Since FenestRA is now available as a Python package on PyPI, you can install it directly using pip:
```bash
pip install napari-fenestra

# To update an existing installation to the latest version, run:
pip install --upgrade napari-fenestra
```

> [!IMPORTANT]
> **Step 2 is still not optional.** As of 0.3.0 the package does declare `AFMReader` (and the
> `pySPM<0.6.3` pin it needs), so a bare `pip install napari-fenestra` no longer leaves you without
> a `.jpk` reader. Two gaps remain by design:
>
> - **`napari` is not declared.** By convention a napari plugin does not depend on napari, because
>   the viewer and its Qt backend are the user's choice. You still install it yourself, in step 2.
> - **`torch` is not declared.** It arrives through `cellpose`, but as the default PyPI wheel rather
>   than the CUDA 12.4 build from step 2. Declaring it here would not change which build you get,
>   and the order in step 2 is what secures GPU acceleration.

### 4. Setup the Deep Learning Backend (Docker vs Singularity)

FenestRA runs its massive deep learning architectures completely independently from the modern Napari UI. You must compile the container engine based on your Operating System.

> [!NOTE]
> **If you built the Docker image before this fix, rebuild it.** `containers/Dockerfile` used to
> end with `ENTRYPOINT ["python"]`. The plugin also passes `python` as the first element of the
> command, and Docker concatenates ENTRYPOINT with that command, so the container ran
> `python python /opt/dl_project/scripts/inference.py` and exited with:
>
> ```text
> can't open file '/opt/python': [Errno 2] No such file or directory
> ```
>
> The `ENTRYPOINT` line has been removed, so a freshly built image is correct. An image built from
> an older checkout still carries the bad `ENTRYPOINT` — `docker build` again to pick up the fix.
>
> The Apptainer / Singularity path was never affected: `singularity exec` bypasses the container's
> `%runscript`, so the explicit `python` is required there. That is why the two engines need
> different argv, and why only one of the two recipes had to change.

First, clone the repository to download the Docker and Singularity setup files:
```bash
git clone https://github.com/LIVR-VUB/FenestRA.git
cd FenestRA
```

**For Windows & macOS Users (Docker Desktop):**
Because Apple and Windows systems cannot securely install Singularity, we use Docker.

<details>
<summary><b>Windows — prerequisites, do these before you build</b></summary>

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Turn on the **WSL 2 backend**: *Settings → General → "Use the WSL 2 based engine"*.
   GPU passthrough only exists on the WSL 2 backend. The Hyper-V backend cannot expose an NVIDIA
   GPU at all, and `--gpus all` — which FenestRA always passes — will fail on it.
3. Install or update the **NVIDIA driver on Windows itself**. Do **not** install a driver or the
   CUDA toolkit inside WSL; the Windows driver is what publishes the GPU into WSL, and installing
   a second one inside the distro breaks it.
4. If you launch napari from inside a WSL distro rather than from Windows, enable that distro
   under *Settings → Resources → WSL Integration*.
5. Prove the GPU is actually reaching containers **before** you spend an hour building:

   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```

   This must print your GPU table. If it prints
   `could not select device driver "" with capabilities: [[gpu]]`, the passthrough is not
   configured, and the plugin's **Run Upsampling** will fail on exactly the same flag.

6. **Disk and time.** The base image is `nvcr.io/nvidia/pytorch:23.01-py3` and it is several
   gigabytes before any of the recipe's own layers. Give Docker Desktop enough disk
   (*Settings → Resources*) and expect the first build to take a long time. Later builds reuse
   the layer cache.

7. **Drive sharing.** The plugin bind-mounts four host directories into the container: your
   Windows temp directory, the output directory, the folder holding your `.pth` model, and the
   installed `fenestra/backend/` directory inside site-packages. On the WSL 2 backend these are
   shared for you. On Hyper-V you must share each drive explicitly, and a model file sitting on a
   drive you have not shared will mount as an empty directory rather than raise an error.
</details>

Build the backend image (Windows and macOS users do **not** need `sudo`):
```bash
cd containers
docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
```
The trailing `..` is the build context, so run the command from inside `containers/`.
`livrvub/dl-upsampling:latest` is only a **local tag** — it is not on Docker Hub and `docker pull`
will not find it. The plugin's Engine field is pre-filled with that exact string so the two line up
once you have built it.

*(In Napari, select **Docker** from the Engine dropdown. No file browsing needed!)* The dropdown
defaults to **Singularity**, so Windows and macOS users must switch it every time the widget is
opened, and switching it back re-fills the path box with a Linux developer path.

> [!NOTE]
> **macOS has no GPU path.** Docker Desktop for macOS has no NVIDIA passthrough, so the `--gpus
> all` that FenestRA always passes has no hardware to expose and the run is expected to fail at
> that flag. The CLAHE (CPU) upsampling method does not launch a container and is unaffected.

**For Native Linux Users (Singularity / Apptainer):**
Linux systems heavily restrict Docker permissions. For ultimate performance and hassle-free paths on Linux, use Apptainer/Singularity.
1. Install Apptainer natively on your Linux distribution.
2. Open a terminal and build the container using the provided definition recipe:
```bash
sudo apptainer build dl_upsampling.sif containers/dl_upsampling.def
```
*(In Napari, select **Singularity** from the Engine dropdown, and use the `...` button to select that `.sif` file!)*

---

## Usage

### Single Image Analysis

1. Activate your environment: `conda activate fenestra-env`
2. Launch napari: `napari`
3. Navigate to `Plugins > FenestRA Pipeline` to open the widget!
4. **Step 1 — Input Data:** Load your `*.jpk-qi-image` file.
5. **Step 2 — Upsampling:** Select a method (CLAHE, HAT, or SwinIR). For DL methods, specify the model `.pth`, choose your Engine (Docker or Singularity), and optionally enable **"Apply Post-DL Sharpening"** with adjustable Clip Limit and Unsharp parameters. Hit **Run Upsampling**.
6. **Step 3 — Segmentation:** Configure Cellpose parameters (Diameter, Cellprob Threshold, Flow Threshold). Optionally load a custom Cellpose model. Hit **Run Cellpose**.
7. **Step 4 — Layout & Analysis:** Click **Arrange 4-Pane Grid** for a synchronized review of Raw, Upsampled, Mask, and Overlay views. Click **Quantify Fenestrations** to export your CSV metrics.

### Batch Analysis

1. Configure your preferred upsampling method, model paths, and Cellpose parameters using the single-image sections above.
2. Scroll down to **Section 5 — Batch Analysis**.
3. Select an **Input Directory** containing your `.jpk-qi-image` files.
4. Select an **Output Directory** where results will be saved.
5. Click **Run Batch**. The status label will update in real-time showing progress (e.g., `Processing 3/10: sample.jpk-qi-image`).
6. When complete, the output directory will contain:
   - `batch_results.xlsx` — Consolidated Excel file with metrics from all images (with `Image_Name` column).
   - `<image_name>_upsampled.tif` — Upsampled TIFF for each input image.
   - `<image_name>_mask.tif` — Cellpose segmentation mask for each input image.

---

## Documentation

A full handbook lives in [`docs/`](docs/) and builds into a searchable MkDocs Material site
covering installation, a screenshot-led walkthrough of all five panels, a parameter and output
reference, the pixel-to-nanometre arithmetic, and the scientific caveats that affect what the
measurements support.

The site builds inside its own small CPU-only container, fully isolated from the DL/GPU stack:

```bash
# Build the docs image once (~200 MB, no CUDA or torch)
apptainer build website/docs.sif website/docs.def

# Live preview at http://127.0.0.1:8000 (local only, nothing is published)
bash website/serve.sh

# Or render the static site into ./site
bash website/build.sh
```

Both `site/` and `website/docs.sif` are gitignored build artefacts.

---

## Changelog

### v0.3
- **All-in-one container.** `containers/Dockerfile.allinone` builds a single image holding napari,
  the plugin, Cellpose and the super-resolution backend, served to a browser over noVNC. On Windows
  and macOS, Docker Desktop becomes the only prerequisite. Launch with `containers/run_fenestra.bat`
  or `containers/run_fenestra.sh`, then open <http://localhost:6080>.
- **New `Local (bundled)` DL engine.** Runs the super-resolution step in a second Python environment
  on the same filesystem instead of launching a container, which is what makes a single image
  possible — a container cannot start a container. `PYTHONPATH` and `PYTHONHOME` are scrubbed from
  that subprocess so the GUI environment cannot leak into the deep-learning one.
- **One container argv builder.** `_build_dl_cmd()` now serves both the interactive worker and the
  batch loop. The two hand-copied argv blocks that had to be kept in sync by hand are gone.
- **No hardcoded developer paths.** The DL model and container fields defaulted to a maintainer's
  home directory in the published package. They now read `FENESTRA_DL_MODEL`, `FENESTRA_SIF`,
  `FENESTRA_DOCKER_IMAGE` and `FENESTRA_ENGINE`, and switching engines no longer overwrites what
  you typed.
- **Cellpose labels now describe Cellpose 4.** The model box said "Leave empty for cyto2" while
  cellpose >= 4.0.1 ignores `model_type` and loads `cpsam`; the diameter box said "0 = auto" when
  cellpose 4 has no auto mode and treats 0 and 30 identically. Both corrected, and the ignored
  `model_type="cyto2"` argument removed. Behaviour is unchanged — only the claims were wrong.
- **`AFMReader` from PyPI.** Git is no longer a prerequisite on Windows.
- **`fenestra.__version__` no longer lies.** It read `0.0.1` in every release; it now reports the
  installed package version.
- **First test.** `tests/test_dl_cmd.py`, plain asserts, no framework.

### v0.2
- **Batch Analysis Module:** New Section 5 in the Napari UI for processing entire folders of `.jpk-qi-image` files. Outputs a single consolidated `.xlsx` Excel file with fenestration metrics from all images, plus individual upsampled TIFFs and Cellpose mask TIFFs.
- **Post-DL Image Enhancement:** Added an optional "Apply Post-DL Sharpening" checkbox that applies CLAHE contrast equalization and Unsharp Masking to the Deep Learning output before Cellpose segmentation.
- **UI Restructuring:** Separated the Clip Limit / Unsharp Radius / Amount sliders into a shared post-processing group that is dynamically visible for both CLAHE and DL workflows.

### v0.1
- **Cross-Platform Docker Support:** Added a `Dockerfile` mirroring the Singularity `.def` environment. Users can now toggle between Docker and Singularity engines directly from the Napari UI.
- **Engine Toggle UI:** New "Engine" dropdown in the Upsampling section. Selecting Docker shows a tag input; selecting Singularity shows a `.sif` file picker.
- **Container Recipes:** Both `Dockerfile` and `dl_upsampling.def` are now bundled in the `containers/` directory.
- **Cross-Platform README:** Added installation instructions for Windows, macOS, and Linux users.

---

## Acknowledgments & Citations

If you use FenestRA in your research, please ensure you properly cite the core technologies that make this pipeline possible:

- **Cellpose** (Instance Segmentation Engine):
  > Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106. https://doi.org/10.1038/s41592-020-01018-x
- **AFMReader** (JPK File Ingestion):
  > Our native support for `.jpk-qi-image` AFM files is powered by the [AFMReader library](https://github.com/AFM-SPM/AFMReader) maintained by the AFM-SPM community.
- **HAT / SwinIR** (Generative Deep Learning Models):
  > Chen, X. et al. (2023). Activating More Pixels in Image Super-Resolution Transformer. (HAT)
  > Liang, J. et al. (2021). SwinIR: Image Restoration Using Swin Transformer. 

---

<img src="https://upload.wikimedia.org/wikipedia/commons/b/b7/Flag_of_Europe.svg" width="50" alt="EU Flag"> 

*This project has received funding from the European Union’s Horizon research and innovation programme under the Marie Skłodowska-Curie grant agreement No 101119613, as part of the [ImAge-d MSCA Doctoral network](https://uit.no/research/image-d).*
