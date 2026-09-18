# Citing FenestRA

What to cite if FenestRA contributed to a published result, and what to write in your methods
section so a reviewer can tell what you actually ran.

## The plugin

Cite the software release by its Zenodo DOI:

> Sarkar, A. (2026). *FenestRA: A napari plugin for LSEC AFM super-resolution and fenestration
> analysis* (Version 0.3.0) [Computer software]. Zenodo.
> <https://doi.org/10.5281/zenodo.19700659>

```bibtex
@software{sarkar_fenestra_2026,
  author    = {Sarkar, Arkajyoti},
  title     = {{FenestRA}: A napari plugin for {LSEC} {AFM} super-resolution
               and fenestration analysis},
  year      = {2026},
  version   = {0.3.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19700659},
  url       = {https://github.com/LIVR-VUB/FenestRA},
  license   = {BSD-3-Clause}
}
```

Replace the version with the one you ran. On a conda + pip install, either `pip show
napari-fenestra` or `python -c "import fenestra; print(fenestra.__version__)"` gives it: since
0.3.0 `__version__` is read back from the installed package metadata, so the two agree and either
is safe to quote.

The all-in-one desktop has no terminal and no napari console, so read the version from the host
instead. While FenestRA is running (the launchers name the container `fenestra`):

```bash
docker exec fenestra /opt/venv-gui/bin/pip show napari-fenestra cellpose
```

If it is not running, ask the image directly — `livrvub/fenestra:latest` for the standard image,
`livrvub/fenestra:cu128` for the Blackwell one:

```bash
docker run --rm --entrypoint /opt/venv-gui/bin/pip livrvub/fenestra:latest show napari-fenestra
```

The image tag alone does not pin the plugin version: the recipe copies the working tree in with
`COPY . /src/fenestra`, so two builds of the same tag can carry different plugin code. Report the
tag **and** the date you pulled or built it alongside the version.

!!! note "On 0.2.11 and earlier"

    `fenestra.__version__` was a hardcoded `"0.0.1"` that never tracked the installed version. If
    your run predates 0.3.0, take the version from `pip show napari-fenestra` and not from the
    package attribute.

!!! note "The manuscript"

    The peer-reviewed manuscript describing this pipeline and its trained models is in
    preparation. Once it is published, cite it alongside the software DOI. The deep-learning
    weights are released with it, not before. See
    [Model weights](../install/model-weights.md).

## Tools FenestRA builds on

These do the segmentation, the file reading and the super-resolution. Cite whichever ones your
run actually used.

**Cellpose** (instance segmentation engine, used in every run):

> Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist
> algorithm for cellular segmentation. *Nature Methods*, 18(1), 100-106.
> <https://doi.org/10.1038/s41592-020-01018-x>

**AFMReader** (JPK file ingestion, used in every run):

> Native support for `.jpk-qi-image` AFM files is powered by the
> [AFMReader library](https://github.com/AFM-SPM/AFMReader) maintained by the AFM-SPM community.

**HAT** (used only if you chose the HAT method):

> Chen, X. et al. (2023). Activating More Pixels in Image Super-Resolution Transformer.

**SwinIR** (used only if you chose the SwinIR method):

> Liang, J. et al. (2021). SwinIR: Image Restoration Using Swin Transformer.

Also cite [napari](https://napari.org) and [scikit-image](https://scikit-image.org), which
provide the viewer and the `regionprops` measurements behind every number in the output table.

## What to put in your methods section

A reviewer needs to be able to reconstruct the measurement from your text alone. Seven things
determine the numbers: the plugin version, the upsampling method and factor, **which container ran
the super-resolution**, whether post-DL sharpening was applied, the Cellpose model, the Cellpose
parameters, and the acquisition scale in nm/px. The acquisition scale matters most, because the
pixel size sets the physical meaning of every diameter in the table and is never checked by the
plugin.

A template, with the values from a typical HAT run substituted:

```text
AFM images of liver sinusoidal endothelial cells were acquired at 100 nm/px and
analyzed with FenestRA (napari-fenestra v0.3.0, DOI 10.5281/zenodo.19700659).
Height-trace channels were read from the raw .jpk-qi-image files with AFMReader
and upsampled x4 with the HAT super-resolution model executed inside the
reference Apptainer container (nvcr.io/nvidia/pytorch:23.01-py3, torch 1.14);
post-DL sharpening was not applied. Fenestrations were segmented with
Cellpose 4.1.1 using the default cpsam model, diameter 30.0,
cellprob threshold 0.00, flow threshold 0.40, min_size 15 px, and per-image
percentile normalization (1.0, 99.0). Area, perimeter, equivalent diameter and
eccentricity were computed with scikit-image regionprops on the resulting
25 nm/px grid; porosity is the summed fenestration area divided by the total
image area.
```

Where each value comes from:

| What to report | Where to read it |
|---|---|
| Plugin version | Conda + pip install: `pip show napari-fenestra`, or `fenestra.__version__` on 0.3.0 and later. All-in-one image: `docker exec fenestra /opt/venv-gui/bin/pip show napari-fenestra` on the host, plus the image tag and build date |
| Acquisition scale | The `Scale: <x> nm/px` line in panel 1 after loading the file |
| Upsampling method and factor | **Method** and **Factor** in panel 2. HAT and SwinIR are always ×4; **Factor** applies to CLAHE only |
| Super-resolution environment | **Engine** in panel 2. `Singularity` or `Docker` (conda + pip install) runs the reference container — report the `.sif` path or the image tag shown in panel 2. `Local (bundled)` (either all-in-one image) runs a second Python environment inside the container; the interpreter is `/opt/venv-dl/bin/python` in **both** all-in-one variants, so report the **image tag** instead. The launcher prints it at startup (`Image:            livrvub/fenestra:latest` from `run_fenestra.bat`, `  image:       livrvub/fenestra:latest` from `run_fenestra.sh`), or run `docker ps --format '{{.Image}}'` on the host while FenestRA is open |
| Post-DL sharpening | The **Apply Post-DL Sharpening** checkbox, and if ticked, the Clip Limit, Unsharp Radius and Unsharp Amount values |
| Cellpose model | The **CP Model** field. Leaving it empty gives `cpsam`, the Cellpose 4 built-in default. Never cyto2: Cellpose 4 ignores `model_type` entirely. Report `cpsam`, or the path of the custom `.pth` you supplied |
| Cellpose parameters | **Diameter**, **Cellprob Thresh** and **Flow Thresh** in panel 3, plus `min_size=15` and the `(1.0, 99.0)` percentile normalization, which are fixed in the code and not exposed in the UI. Under Cellpose 4 the diameter is a rescaling factor (`30 / diameter`), so `30` and `0` both mean *no rescale*; there is no automatic estimate to report |
| Cellpose version | Conda + pip install: `pip show cellpose`. All-in-one image: `docker exec fenestra /opt/venv-gui/bin/pip show cellpose` on the host |

Every control and its default is listed in [Parameters](../reference/parameters.md); the
arithmetic that turns pixels into nanometers is in [Metrics](../reference/metrics.md).

!!! warning

    State the acquisition scale explicitly. The super-resolution models were trained to invert a
    ×4 degradation of scans acquired near 100 nm/px, and the plugin never checks the pixel size
    of what you feed it. A scan acquired far from that band produces a plausible-looking image
    and out-of-domain measurements. Read
    [The scale-domain question](../caveats/scale-domain.md) before writing the sentence.

!!! warning "If you ran an all-in-one image, name the tag"

    Neither [all-in-one image](../install/all-in-one.md) carries the reference deep-learning
    stack, and the two differ from each other, so "the all-in-one image" is not a stack a reviewer
    can reconstruct:

    - **`livrvub/fenestra:latest`** (`containers/Dockerfile.allinone`) — GUI venv on
      **torch 2.4.0+cu124**, `/opt/venv-dl` on **torch 2.1.2 / torchvision 0.16.2**. The
      super-resolution ran on 2.1.2, the Cellpose segmentation on 2.4.0.
    - **`livrvub/fenestra:cu128`** (`containers/Dockerfile.allinone.cu128`, required for RTX
      50-series) — **torch 2.8.0 / torchvision 0.23.0** in *both* venvs, so the super-resolution
      and the Cellpose segmentation both ran on 2.8.0. basicsr is patched here from
      `torchvision.transforms.functional_tensor` to `torchvision.transforms.functional`, because
      0.17 deleted that module.

    Neither is the **torch 1.14** on `nvcr.io/nvidia/pytorch:23.01-py3` that
    `containers/dl_upsampling.def` builds and that the method was developed against. The weights,
    the architectures and the arithmetic are the same, but the three stacks are not bit-for-bit
    equivalent. The two images are indistinguishable from inside the desktop, so name the exact
    tag and its torch version — see
    [The two images, compared](../install/all-in-one.md#the-two-images-compared), which is the
    source of truth for these versions.

    Where the hardware allows it, numbers intended for publication should come from the reference
    container. On Blackwell (RTX 50-series, sm_120) it does not: torch 1.14 carries no sm_120
    kernels and will not run there at all, so a Blackwell result can only come from
    `livrvub/fenestra:cu128` and the methods section has to say so.
