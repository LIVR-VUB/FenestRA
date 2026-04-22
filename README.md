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
[![run with docker](https://img.shields.io/badge/run%20with-docker-0db7ed.svg?labelColor=000000&logo=docker)](https://www.docker.com/)
[![run with apptainer/singularity](https://img.shields.io/badge/run%20with-apptainer%2Fsingularity-1E95D3.svg?labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMjQ1IiBoZWlnaHQ9IjI0MCIgdmlld0JveD0iNjAgMCAzMTAgMjUwIiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgk8cGF0aCBkPSJtIDI3MC4xOCwyNTMuOTggYyAtMS44LC0xLjIgLTMuNCwtMyAtNC40LC01LjIgbCAtNTIuNiwtMTE3LjQgYyAtMi4yLC00LjggLTMuOCwtOC42IC01LjIsLTExLjYgLTIuMiwtNC40IC0yLjIsLTUuNiAtMi4yLC02LjQgMCwtMi4yIDAuOCwtMy44IDIuNiwtNC44IHYgLTQuNCBoIC00My4yIHYgNC40IGMgMC44LDAuNCAxLjIsMS4yIDEuOCwxLjggMC40LDAuOCAwLjgsMS44IDAuOCwzIDAsMS4yIC0wLjQsMyAtMS44LDUuNiAtMS4yLDIuNiAtMi42LDUuNiAtNC40LDkuNCBsIC01MS44LDExNyBjIC0wLjgsMS44IC0yLjIsNC40IC0zLjgsNy40IC0xLjgsMyAtNC44LDQuNCAtOC4yLDQuOCB2IDMuOCBoIDQ5LjYgdiAtMy44IGMgLTUuNiwwIC04LjIsLTIuMiAtOC4yLC01LjYgMCwtMS44IDAuOCwtNC44IDMsLTkgMS44LC0zLjQgMy44LC03LjggNS42LC0xMiAyNC42LDkuNCA1Mi4yLDEwIDc2LjgsMC44IDIuMiw0LjQgMy44LDguMiA1LjIsMTEuMiAxLjgsMy40IDIuNiw2LjQgMi42LDguNiAwLDIuMiAtMC48LDMuOCAtMi4yLDQuOCAtMS4yLDAuNCAtMi4yLDAuOCAtMy40LDEuMiB2IDMuOCBoIDUwLjQgdiAtMy44IGMgLTIuOCwtMS44IC01LjQsLTIuOCAtNywtMy42IHogbSAtMTExLjQsLTQ3IDI3LjYsLTYxLjQgMjgsNjIuMiBjIC0xOCw2IC0zNy40LDYgLTU1LjYsLTAuOCB6IiBmaWxsPSJ3aGl0ZSIvPiA8cGF0aCBkPSJtIDg5Ljc4LDE0MC45OCBjIDAsLTkgMS4yLC0xNy42IDMuNCwtMjYuNCBsIC0yOCwtMTIuNiBjIC0zLjgsMTIgLTYsMjQuNiAtNiwzNy42IDAsMzUgMTQuMiw2OC42IDM5LjgsOTIuOCBsIDEuOCwtMy40IDExLjIsLTI1LjQgYyAtMTMuNiwtMTcuNCAtMjIuMiwtMzkgLTIyLjIsLTYyLjYgeiIgZmlsbD0iIzkzOTU5OCIvPiA8cGF0aCBkPSJtIDMxMC4xOCwxMDIuNTggLTI4LDEyLjYgYyAyLjIsOC4yIDMuNCwxNi44IDMuNCwyNS44IDAsMjMuOCAtOC42LDQ1LjggLTIyLjgsNjIuNiBsIDExLjYsMjUuNCAxLjgsMy40IGMgMjUuNCwtMjQuMiAzOS44LC01Ny44IDM5LjgsLTkyLjggLTAuMiwtMTIuNCAtMi4yLC0yNSAtNS44LC0zNyB6IiBmaWxsPSIjRjc5NDIxIi8%2BIDxwYXRoIGQ9Im0gNzEuMTgsODYuOTggMjcuNiwxMi42IGMgMTQuNiwtMzEgNDQuOCwtNTMgODAuMiwtNTYuMiB2IC0zMC42IGMgLTQ2LDIuNiAtODguNCwzMS40IC0xMDcuOCw3NC4yIHoiIGZpbGw9IiMxRTk1RDMiLz4gPHBhdGggZD0ibSAzMDQuMTgsODYuOTggYyAtMTkuNCwtNDIuOCAtNjEuOCwtNzEuNiAtMTA4LjQsLTc0LjYgdiAzMC42IGMgMzUuOCwzIDY2LDI1IDgwLjYsNTYuMiB6IiBmaWxsPSIjNkZCNTQ0Ii8%2BPC9zdmc%2B)](https://sylabs.io/docs/)

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

### 1. Requirements
- Python 3.10+
- An NVIDIA GPU with CUDA 12.4 drivers (recommended for DL inference)
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

# Install Cellpose for fenestration instance segmentation
pip install cellpose

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

### 4. Setup the Deep Learning Backend (Docker vs Singularity)

FenestRA runs its massive deep learning architectures completely independently from the modern Napari UI. You must compile the container engine based on your Operating System.

First, clone the repository to download the Docker and Singularity setup files:
```bash
git clone https://github.com/LIVR-VUB/FenestRA.git
cd FenestRA
```

**For Windows & macOS Users (Docker Desktop):**
Because Apple and Windows systems cannot securely install Singularity, we use Docker.
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) on your machine.
2. Open a terminal and navigate to this repository's `containers/` directory.
3. Build the backend image (Windows/Mac users do NOT need `sudo`):
```bash
docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
```
*(In Napari, select **Docker** from the Engine dropdown. No file browsing needed!)*

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
