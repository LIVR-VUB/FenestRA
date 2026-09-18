# Parameters

Every control in the FenestRA dock, panel by panel, with its default and its real behavior. Since
0.3.0 the path defaults come from environment variables rather than from anyone's home directory;
they are listed in [Environment variables](#environment-variables). A final section lists the values
that are fixed in code and cannot be reached from the interface.

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
| `DL Model:` + `...` | empty, or `$FENESTRA_DL_MODEL`; placeholder `Path to the .pth checkpoint` | File picker, filter `*.pth` | Path to the super-resolution checkpoint. Under Singularity and Docker its directory is bind-mounted into the container; under `Local (bundled)` it is passed through as a real path | Empty until you point it at a checkpoint. **Before 0.3.0 this box was pre-filled with a developer path** under `/home/arka/`, which existed on no other machine. The weights are not public; see [Model weights](../install/model-weights.md). |
| `Engine:` | `Singularity`, or `$FENESTRA_ENGINE` | `Singularity`, `Docker`, `Local (bundled)` | Which backend runs the network. Singularity and Docker launch a container; **Local (bundled)** runs `inference.py` under a second Python interpreter on the same filesystem, as a plain subprocess with no bind mounts and no path translation | Linux hosts use Singularity/Apptainer. `Local (bundled)` is new in 0.3.0 and exists for the [all-in-one container](../install/all-in-one.md), which sets `FENESTRA_ENGINE=Local`: a container cannot start a container. Elsewhere it refuses with a named path unless `FENESTRA_DL_PYTHON` points at a suitable interpreter. See [Container backend](../install/container-backend.md). |
| `Apply Post-DL Sharpening` | unchecked | on / off | Runs CLAHE plus unsharp masking on the network output | **Changes both the data type and the units of the saved image.** See [Output files](outputs.md). |
| `Singularity (.sif):` / `Docker Tag:` / `DL backend:` + `...` | empty, or `$FENESTRA_SIF`, under Singularity; `livrvub/dl-upsampling:latest`, or `$FENESTRA_DOCKER_IMAGE`, under Docker; unused under Local | File picker, filter `*.sif *.def` (Singularity only) | The container image to run | The label, the placeholder and the contents follow the Engine dropdown. **Each engine keeps its own value**: switch to Docker and back and the Singularity path you typed is still there. Before 0.3.0, switching back overwrote it with a developer path. Under `Local (bundled)` the field is disabled and shows only which interpreter will be used. |
| `Run Upsampling` | — | — | Runs the chosen route and adds the layer **Upsampled AFM** | The CLAHE route runs on the GUI thread, so napari stops responding until it finishes. It is not crashed. |

### Which controls are visible

| Method | Factor | Clip Limit, Unsharp Radius, Unsharp Amount | DL Model, Engine, container, sharpening checkbox |
|---|---|---|---|
| `CLAHE (CPU)` | shown | shown | hidden |
| `HAT` / `SwinIR`, sharpening off | hidden | hidden | shown |
| `HAT` / `SwinIR`, sharpening on | hidden | shown | shown |

Under `Local (bundled)` the container row is still shown, but the text field is disabled and its
`...` button is hidden, because there is no image to choose.

!!! warning "The bundled Local environment is not the reference stack"

    Inside the [all-in-one container](../install/all-in-one.md) the `Local (bundled)` engine runs
    **torch 2.1.2 / torchvision 0.16.2**, not the reference stack's torch 1.14 on
    `nvcr.io/nvidia/pytorch:23.01-py3`. Torch 1.13 and 1.14 wheels carry no PTX and will not start
    on any GPU newer than sm_86, which rules out the RTX 40-series and the H100, so the all-in-one
    image could not use them. `containers/dl_upsampling.def` still builds the reference stack and
    is unchanged. **Numbers intended for publication should come from the reference container.**

## Panel 3. Cellpose Segmentation

| Control | Default | Range or options | What it does | Notes |
|---|---|---|---|---|
| `CP Model:` + `...` | empty, or `$FENESTRA_CP_MODEL`; placeholder `Leave empty for the Cellpose 4 default (cpsam)` | Any file path | Path to a Cellpose checkpoint. If the path does not exist on disk, the plugin builds `CellposeModel` with no pretrained model, which loads the Cellpose 4 default, `cpsam` | The placeholder read `Leave empty for cyto2` before 0.3.0. **The label was wrong, not the behaviour**: an empty box has always given you `cpsam`. |
| `Diameter (30 = no rescale):` | `30.00` | 0.00 to 500.00, step 1.00 | Sets Cellpose 4's image rescaling as `30 / diameter` | **Not a size in pixels, and there is no auto mode.** `0` and `30` both mean "no rescaling" — `0` fails Cellpose's `> 0` test. Below 30 the image is upscaled before segmentation, above 30 it is downscaled. The label read `Diameter (0=auto):` before 0.3.0. |
| `Cellprob Thresh:` | `0.00` | −10.00 to 10.00, step 0.1 | Cell probability threshold. Lower values accept more, and larger, pores | Behaves as documented by Cellpose. |
| `Flow Thresh:` | `0.40` | 0.00 to 10.00, step 0.1 | Maximum allowed flow error per mask. Lower values reject more irregular shapes | Behaves as documented by Cellpose. |
| `Run Cellpose` | — | — | Segments the upsampled image on the host GPU and adds the labels layer **Cellpose Masks** | Runs on the upsampled image, never on the raw one. |

!!! warning "0.3.0 corrected these two labels. It did not change what they do."

    Up to 0.2.11 the plugin passed `model_type="cyto2"` when **CP Model** was empty. Cellpose 4.0.1
    and later accept that argument, log `model_type argument is not used in v4.0.1+. Ignoring this
    argument...`, and load `cpsam` anyway. 0.3.0 drops the argument and rewrites the placeholder, so
    the interface now states what the code always did: **you get `cpsam`**. `setup.cfg` pins
    `cellpose>=4.0.1` as a hard floor for the same reason. Masks are unchanged across the two
    versions.

    `Diameter` is still not a size in pixels. Cellpose 4 computes `image_scaling = 30. / diameter`,
    so the default of 30 is a no-op, and 0 fails the `> 0` test and is also a no-op. Setting it to
    15 doubles the image before segmentation; setting it to 60 halves it. Only the label changed.

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

## Environment variables

New in 0.3.0. These set the defaults the dock opens with, so one wheel is correct on a developer
box, on an HPC node and inside the all-in-one image. They are read once, when the widget is built,
so export them before starting napari. Nothing here overrides a value you type: the interface wins
for the rest of the session.

| Variable | Sets | Default if unset |
|---|---|---|
| `FENESTRA_ENGINE` | The engine selected at startup. Matched on the first word only, so `Local` selects `Local (bundled)` | `Singularity` |
| `FENESTRA_DL_MODEL` | **DL Model** | empty |
| `FENESTRA_SIF` | The container box while the engine is Singularity | empty |
| `FENESTRA_DOCKER_IMAGE` | The container box while the engine is Docker | `livrvub/dl-upsampling:latest` |
| `FENESTRA_CP_MODEL` | **CP Model** | empty |
| `FENESTRA_DL_PYTHON` | The interpreter the `Local (bundled)` engine runs `inference.py` with. There is no control for this in the interface | `/opt/venv-dl/bin/python` |

The all-in-one image sets `FENESTRA_ENGINE`, `FENESTRA_DL_PYTHON` and `FENESTRA_DL_MODEL` in its
Dockerfile, which is why its dock opens ready to run.

!!! note "The Local engine gets a cleaned environment"

    Everything else is inherited, but `PYTHONPATH` and `PYTHONHOME` are stripped from the
    subprocess. Either one, set for the napari environment, would drag that environment's numpy and
    torch into the deep-learning interpreter, which is the one thing two separate environments exist
    to prevent.

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

Since 0.3.0 all three engines get their command line from one builder, `_build_dl_cmd()` in
`pipeline.py`, used by both the interactive run and the batch loop. The two hand-copied argv blocks
that could drift apart are gone. `tests/test_dl_cmd.py` checks the builder with six
plain assertions; run it from the FenestRA environment with `python tests/test_dl_cmd.py`.

### Other

| Setting | Value | Consequence |
|---|---|---|
| JPK channel | `height_trace` | No other channel can be selected. |
| JPK flip | `flip_image=True` | The image is flipped vertically on load. |
| CLAHE bins | `256` | Passed to `equalize_adapthist`. |
| CLAHE zoom order | `3` (cubic spline) | Used by the CLAHE route before contrast equalization. |
| Layer scale | `(0.25, 0.25)` | Applied to the **Upsampled AFM**, **Cellpose Masks** and **Overlay** layers. Correct for ×4, wrong for any other CLAHE factor. |
