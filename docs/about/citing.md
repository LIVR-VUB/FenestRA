# Citing FenestRA

What to cite if FenestRA contributed to a published result, and what to write in your methods
section so a reviewer can tell what you actually ran.

## The plugin

Cite the software release by its Zenodo DOI:

> Sarkar, A. (2026). *FenestRA: A napari plugin for LSEC AFM super-resolution and fenestration
> analysis* (Version 0.2.11) [Computer software]. Zenodo.
> <https://doi.org/10.5281/zenodo.19700659>

```bibtex
@software{sarkar_fenestra_2026,
  author    = {Sarkar, Arkajyoti},
  title     = {{FenestRA}: A napari plugin for {LSEC} {AFM} super-resolution
               and fenestration analysis},
  year      = {2026},
  version   = {0.2.11},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19700659},
  url       = {https://github.com/LIVR-VUB/FenestRA},
  license   = {BSD-3-Clause}
}
```

Replace the version with the one you ran, from `pip show napari-fenestra`. `fenestra.__version__`
reports `0.0.1` regardless of what is installed, so do not quote it.

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

A reviewer needs to be able to reconstruct the measurement from your text alone. Six things
determine the numbers: the plugin version, the upsampling method and factor, whether post-DL
sharpening was applied, the Cellpose model, the Cellpose parameters, and the acquisition scale in
nm/px. The last one matters most, because the pixel size sets the physical meaning of every
diameter in the table and is never checked by the plugin.

A template, with the values from a typical HAT run substituted:

```text
AFM images of liver sinusoidal endothelial cells were acquired at 100 nm/px and
analyzed with FenestRA (napari-fenestra v0.2.11, DOI 10.5281/zenodo.19700659).
Height-trace channels were read from the raw .jpk-qi-image files with AFMReader
and upsampled x4 with the HAT super-resolution model executed inside an
Apptainer container; post-DL sharpening was not applied. Fenestrations were
segmented with Cellpose 4.1.1 using the default cpsam model, diameter 30.0,
cellprob threshold 0.00, flow threshold 0.40, min_size 15 px, and per-image
percentile normalization (1.0, 99.0). Area, perimeter, equivalent diameter and
eccentricity were computed with scikit-image regionprops on the resulting
25 nm/px grid; porosity is the summed fenestration area divided by the total
image area.
```

Where each value comes from:

| What to report | Where to read it |
|---|---|
| Plugin version | `pip show napari-fenestra` |
| Acquisition scale | The `Scale: <x> nm/px` line in panel 1 after loading the file |
| Upsampling method and factor | **Method** and **Factor** in panel 2. HAT and SwinIR are always ×4; **Factor** applies to CLAHE only |
| Post-DL sharpening | The **Apply Post-DL Sharpening** checkbox, and if ticked, the Clip Limit, Unsharp Radius and Unsharp Amount values |
| Cellpose model | The **CP Model** field. Leaving it empty gives `cpsam` under Cellpose 4, not cyto2, whatever the placeholder text says |
| Cellpose parameters | **Diameter**, **Cellprob Thresh** and **Flow Thresh** in panel 3, plus `min_size=15` and the `(1.0, 99.0)` percentile normalization, which are fixed in the code and not exposed in the UI |
| Cellpose version | `pip show cellpose` |

Every control and its default is listed in [Parameters](../reference/parameters.md); the
arithmetic that turns pixels into nanometers is in [Metrics](../reference/metrics.md).

!!! warning

    State the acquisition scale explicitly. The super-resolution models were trained to invert a
    ×4 degradation of scans acquired near 100 nm/px, and the plugin never checks the pixel size
    of what you feed it. A scan acquired far from that band produces a plausible-looking image
    and out-of-domain measurements. Read
    [The scale-domain question](../caveats/scale-domain.md) before writing the sentence.
