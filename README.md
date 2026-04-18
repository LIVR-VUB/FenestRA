<p align="center">
  <img src="misc/FenestRA.jpg" alt="FenestRA Logo" width="450"/>
</p>

# FenestRA
**Fenestration Resolution & Analysis Pipeline**

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Napari](https://img.shields.io/badge/napari-plugin-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg?logo=nvidia)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-ee4c2c.svg?logo=pytorch)

FenestRA is a custom Napari plugin built for the Advanced LSEC AFM Pipeline. It bridges the gap between interactive Napari features, legacy deep-learning upscale repositories via Apptainer, and state-of-the-art Cellpose instance segmentation. 

By combining Deep Learning-based Super Resolution (HAT / SwinIR) with automated morphological analysis, FenestRA drastically simplifies the workflow of extracting robust physical porosity and fenestration morphology metrics directly from raw `.jpk.qi-image` files.

---

## Features
- **Hub-and-Spoke Deep Learning Architecture:** Run legacy Python 3.8 dependent upscale models asynchronously inside a Singularity container without freezing your modern Napari GUI.
- **Native JPK Ingestion:** Automatically reads native physical scale (`nm / px`) from `.jpk.qi-image` files using AFMReader.
- **Synchronized 4-Pane Analysis:** Auto-generates a synchronized Napari viewer layout combining Raw, Upsampled, Mask, and Boundary Overlays natively.
- **Configurable Fallbacks:** Includes high-fidelity Python-based CLAHE and unsharp masking functions when DL inference isn't required.
- **Sub-cellular Quantification:** Automatically calculates standard metrics + digital-to-physical size translations directly to a `.csv`.

---

## Installation

### 1. Requirements
Ensure you have `Apptainer` or `Singularity` installed on your system to run the Super-Resolution container logic.

### 2. Create the Host Environment
Create a clean Anaconda environment optimized for Cellpose targeting CUDA 12.4:

```bash
conda create -n fenestra-env -c conda-forge python=3.10 numpy=1.26.4
conda activate fenestra-env

# Install base GUI tools, Napari, and core scientific dependencies
pip install "napari[all]" magicgui qtpy scipy scikit-image pandas tifffile "numpy<2"

# Install PyTorch mapped explicitly to CUDA 12.4 to ensure GPU hardware acceleration works
pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.4.0 torchvision==0.19.0

# Install Cellpose for fenestration instance segmentation
pip install cellpose

# Install AFMReader for handling raw JPK AFM metadata
pip install git+https://github.com/AFM-SPM/AFMReader.git
```

### 3. Install FenestRA
Clone this repository and install it in "editable" mode:
```bash
git clone https://github.com/LIVR-VUB/FenestRA.git
cd FenestRA
pip install -e .
```

### 4. Setup the Deep Learning Backend (Docker vs Singularity)

FenestRA runs its massive deep learning architectures completely independently from the modern Napari UI. You must compile the container engine based on your Operating System:

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

1. Activate your environment: `conda activate fenestra-env`
2. Launch napari: `napari`
3. Navigate to `Plugins > FenestRA Pipeline` to open the widget!
4. **Step 1:** Load your `*.jpk-qi-image` file.
5. **Step 2:** Select an upsampling method. If using HAT/SwinIR, pick your Model `.pth` and Singularity `.sif` container and hit Run. Wait for the background thread to finish.
6. **Step 3:** Setup Cellpose parameters. If using automatic cluster diameter estimation, set Diameter to `0`. Hit Run Cellpose.
7. **Step 4:** Arrange grid for 4-pane layout reviewing, and click **Quantify Fenestrations** to export your CSV metrics!
