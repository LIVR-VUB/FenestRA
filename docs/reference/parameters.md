# Parameters

Every control in the FenestRA dock, panel by panel, with its default and its real behavior. Two
labels in panel 3 describe an older version of Cellpose and are flagged below. A final section lists
the values that are fixed in code and cannot be reached from the interface.

## Panel 1. Input Data

| Control | Default | Range or options | What it does | Notes |
|---|---|---|---|---|
| `Load JPK.qi-image` | — | File dialog, filter `JPK Files (*.jpk *.jpk-qi-image)` | Reads the scan and its physical scale, adds the layer **Raw AFM** with the `magma` colormap | Always reads the `height_trace` channel, with the image flipped vertically. No channel selector is exposed. |
| Info label | `No file loaded.` | read only | Shows `Size: (H, W)` and `Scale: <x> nm/px` once a file is loaded | The nm/px value is what every measurement is later derived from. Check it. |

!!! warning "The scale is read but never checked"

    The models were trained to invert a ×4 degradation from roughly 100 nm/px. Nothing in the
    plugin compares your scan's nm/px against that band, and nothing will warn you if it falls
    outside. See [The scale-domain question](../caveats/scale-domain.md).

## Panel 2. Upsampling / Enhancement

| Control | Default | Range or options | What it does | Notes |
|---|---|---|---|---|
| `Method:` | `CLAHE (CPU)` | `CLAHE (CPU)`, `HAT`, `SwinIR` | Chooses the upsampling route. CLAHE runs on the host CPU; HAT and SwinIR run in the container on the GPU | Changing this after a run, but before clicking **Quantify Fenestrations**, changes the reported sizes. See [Known issues](../caveats/known-issues.md). |
| `Factor:` | `4` | 1 to 10, integer | Cubic-spline zoom factor for the CLAHE route | Visible for CLAHE only. **It has no effect on HAT or SwinIR**, which are fixed at ×4. Values other than 4 also leave the napari layer scale wrong, so Raw and Upsampled look aligned in the grid view when they are not. |
| `Clip Limit:` | `0.020` | 0.000 to 99.990 (no range set in code, so the spinbox default applies), step 0.01 | CLAHE contrast clip limit, passed to `equalize_adapthist` | Applies to the CLAHE route, and to the DL route only when **Apply Post-DL Sharpening** is ticked. |
| `Unsharp Radius:` | `1.00` | 0.00 to 99.99 (spinbox default), step 1.00 | Gaussian radius of the unsharp mask | Same visibility rule as Clip Limit. Use the keyboard rather than the arrows for small changes. |
| `Unsharp Amount:` | `1.00` | 0.00 to 99.99 (spinbox default), step 1.00 | Strength of the unsharp mask | Same visibility rule as Clip Limit. |
| `DL Model:` + `...` | `/home/arka/Desktop/AFM-Project/DL_Upsampling/models/best_model_ema.pth` | File picker, filter `*.pth` | Path to the super-resolution checkpoint, bind-mounted into the container | The default is a developer path that will not exist on your machine. You must replace it. The weights are not public; see [Model weights](../install/model-weights.md). |
| `Engine:` | `Singularity` | `Singularity`, `Docker` | Which container runtime is invoked | Linux hosts use Singularity/Apptainer. See [Container backend](../install/container-backend.md). |
| `Apply Post-DL Sharpening` | unchecked | on / off | Runs CLAHE plus unsharp masking on the network output | **Changes both the data type and the units of the saved image.** See [Output files](outputs.md). |
| `Singularity (.sif):` / `Docker Tag:` + `...` | `/home/arka/Desktop/AFM-Project/DL_Upsampling/containers/dl_upsampling.sif` | File picker, filter `*.sif *.def` (hidden in Docker mode) | The container image to run | The label and the contents swap with the Engine dropdown. Selecting Docker replaces the text with `livrvub/dl-upsampling:latest`. Selecting Singularity again **overwrites whatever you typed** with the developer path above. |
| `Run Upsampling` | — | — | Runs the chosen route and adds the layer **Upsampled AFM** | The CLAHE route runs on the GUI thread, so napari stops responding until it finishes. It is not crashed. |

### Which controls are visible

| Method | Factor | Clip Limit, Unsharp Radius, Unsharp Amount | DL Model, Engine, container, sharpening checkbox |
|---|---|---|---|
| `CLAHE (CPU)` | shown | shown | hidden |
| `HAT` / `SwinIR`, sharpening off | hidden | hidden | shown |
| `HAT` / `SwinIR`, sharpening on | hidden | shown | shown |

## Panel 3. Cellpose Segmentation

| Control | Default | Range or options | What it does | Notes |
|---|---|---|---|---|
| `CP Model:` + `...` | empty, placeholder `Leave empty for cyto2` | Any file path | Path to a Cellpose checkpoint. If the path does not exist on disk, the plugin falls back to a built-in model | **The placeholder is wrong under Cellpose 4.** Leaving it empty gives you `cpsam`, not `cyto2`. |
| `Diameter (0=auto):` | `30.00` | 0.00 to 500.00, step 1.00 | Sets Cellpose 4's image rescaling as `30 / diameter` | **The label is wrong under Cellpose 4.** There is no auto mode. `0` and `30` both mean "no rescaling". Below 30 the image is upscaled before segmentation, above 30 it is downscaled. |
| `Cellprob Thresh:` | `0.00` | −10.00 to 10.00, step 0.1 | Cell probability threshold. Lower values accept more, and larger, pores | Behaves as documented by Cellpose. |
| `Flow Thresh:` | `0.40` | 0.00 to 10.00, step 0.1 | Maximum allowed flow error per mask. Lower values reject more irregular shapes | Behaves as documented by Cellpose. |
| `Run Cellpose` | — | — | Segments the upsampled image on the host GPU and adds the labels layer **Cellpose Masks** | Runs on the upsampled image, never on the raw one. |

!!! warning "Two labels in this panel describe Cellpose 2"

    The plugin passes `model_type="cyto2"` when **CP Model** is empty. Cellpose 4.0.1 and later
    accept that argument, log `model_type argument is not used in v4.0.1+. Ignoring this
    argument...`, and load `cpsam` instead. You get plausible masks from a different network than
    the label promises.

    `Diameter` is no longer a size in pixels. Cellpose 4 computes `image_scaling = 30. / diameter`,
    so the default of 30 is a no-op, and 0 fails the `> 0` test and is also a no-op. Setting it to
    15 doubles the image before segmentation; setting it to 60 halves it.

    Details and the full list in [Segmentation](../guide/step3-segmentation.md).

## Panel 4. Layout & Analysis

| Control | Default | Range or options | What it does | Notes |
|---|---|---|---|---|
| `Arrange 4-Pane Grid` | — | — | Builds an RGB **Overlay** layer with red pore boundaries drawn on the upsampled image, then switches napari to a 2×2 grid | The Overlay layer is only created when both an upsampled image and a mask exist. Otherwise the button only enables the grid. |
| `Quantify Fenestrations` | — | — | Measures every labeled pore, opens a save dialog defaulting to `fenestration_metrics.csv`, then reports the count and porosity in a message box | The upsampling factor used for the conversion is read from the **Method** dropdown at the moment you click, not from what was actually run. Cancelling the save dialog discards the measurement and shows no message. |

## Panel 5. Batch Analysis

| Control | Default | Range or options | What it does | Notes |
|---|---|---|---|---|
| `Input Dir:` + `Browse` | empty, placeholder `Folder containing .jpk-qi-image files` | Directory picker | Folder to scan | Matches `*.jpk-qi-image` and `*.jpk`, sorted and deduplicated. Not recursive. |
| `Output Dir:` + `Browse` | empty, placeholder `Folder for results (Excel + TIFFs)` | Directory picker | Where results are written | Created if it does not exist. Existing files with the same names are overwritten. |
| `Status:` | `Idle` | read only | Shows `Processing 3/10: sample.jpk-qi-image` while running | Ends at `Complete — N images processed.` |
| `Run Batch` | — | — | Runs load, upsample, segment, and measure for every file found | **Reuses the settings from panels 2 and 3 as they stand.** There is no separate batch configuration. |

See [Batch analysis](../guide/step5-batch.md) for the full workflow.

## Values fixed in code

These are not exposed in the interface. They are listed so you can report them in a methods
section.

### Cellpose

| Setting | Value | Consequence |
|---|---|---|
| `normalize` | `{"normalize": True, "percentile": (1.0, 99.0)}` | Contrast is stretched between the 1st and 99th percentile **of each image separately**. In a batch run every image gets its own stretch, so segmentations are not strictly comparable across rows of one workbook. |
| `min_size` | `15` pixels | Any mask smaller than 15 pixels is discarded. In nanometers this depends entirely on your output grid; see [Metrics & units](metrics.md). |
| `do_3D` | `False` | 2D segmentation only. |
| `augment` | `False` | No test-time augmentation. |
| `channels` | `None` | Single-channel input. |
| `channel_axis` | `None` | Single-channel input. |
| `tile` | `True`, then dropped | Cellpose 4.1.1 has no `tile` argument. The plugin passes it, catches the resulting `TypeError`, removes the argument and retries. Every run therefore makes one failed call before the real one. |

### Deep-learning inference

| Setting | Value | Consequence |
|---|---|---|
| Upscale factor | `4` | Fixed for both HAT and SwinIR. The **Factor** spinbox does not change it. |
| `--tile_size` | `256` | Tiled inference always runs, even on images small enough to fit whole. |
| Tile overlap | `32` pixels | Overlapping tiles are averaged with equal weight, not feathered, so seams are possible. |
| `window_size` | `16` for HAT, `8` for SwinIR | The image is reflect-padded up to a multiple of this before inference, then cropped back. |
| Architecture | `embed_dim=180`, `depths=[6]*6`, `num_heads=[6]*6` | Hardcoded. Checkpoints trained with a different width or depth will not load. |
| `load_state_dict` | `strict=True` | A mismatched checkpoint raises rather than loading partially. This is deliberate: it is what stops a wrong network from producing plausible output. |
| Normalization | per-image min-max to `[0, 1]`, output clipped to `[0, 1]` | Applied inside the container before and after the network. See [The scale-domain question](../caveats/scale-domain.md). |
| Output filename | `<input_stem>_SR4x.tif`, float32 | Written to a temporary directory, then read back by the plugin. |

### Other

| Setting | Value | Consequence |
|---|---|---|
| JPK channel | `height_trace` | No other channel can be selected. |
| JPK flip | `flip_image=True` | The image is flipped vertically on load. |
| CLAHE bins | `256` | Passed to `equalize_adapthist`. |
| CLAHE zoom order | `3` (cubic spline) | Used by the CLAHE route before contrast equalization. |
| Layer scale | `(0.25, 0.25)` | Applied to the **Upsampled AFM**, **Cellpose Masks** and **Overlay** layers. Correct for ×4, wrong for any other CLAHE factor. |
