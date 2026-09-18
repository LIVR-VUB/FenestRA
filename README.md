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

### 1. Requirements
- Python 3.10+
- An NVIDIA GPU with CUDA 12.4 drivers (recommended for DL inference)
- **Git.** One dependency (`AFMReader`) is installed straight from a git repository, so `pip`
  needs a working `git` on your `PATH`. On Windows this is not present by default — install
  [Git for Windows](https://git-scm.com/download/win) first, or step 2 fails with
  `ERROR: Cannot find command 'git'`.

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
pip install "napari[all]" magicgui qtpy scipy scikit-image pandas tifffile "numpy<2" openpyxl

# Install PyTorch mapped explicitly to CUDA 12.4 to ensure GPU hardware acceleration works
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.0 torchvision==0.19.0

# Install Cellpose for fenestration instance segmentation (pinned: the docs and UI labels describe 4.1.1 behavior)
pip install cellpose==4.1.1

# Install AFMReader for handling raw JPK AFM metadata
pip install git+https://github.com/AFM-SPM/AFMReader.git
```

### 3. Install FenestRA
Since FenestRA is now available as a Python package on PyPI, you can install it directly using pip:
```bash
pip install napari-fenestra

# To update an existing installation to the latest version, run:
pip install --upgrade napari-fenestra
```

> [!IMPORTANT]
> **Step 2 is not optional.** The published package does not declare `napari`, `AFMReader`, or
> `torch` in `install_requires`, even though all three are imported at runtime. Installing
> `napari-fenestra` on its own therefore leaves you with no viewer to dock into and no `.jpk`
> reader. `AFMReader` is distributed from git rather than PyPI, which is why it cannot be declared
> as an ordinary dependency. `torch` does arrive indirectly via `cellpose`, but as the default
> PyPI wheel rather than the CUDA 12.4 build from step 2, so you lose GPU acceleration.

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
