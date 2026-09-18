# Output files

Everything FenestRA writes to disk, with exact column names in order. Single-image analysis
produces one CSV. A batch run produces one workbook plus two TIFFs per input scan.

| File | Written by | Where | Format |
|---|---|---|---|
| `fenestration_metrics.csv` | **Quantify Fenestrations**, panel 4 | A path you choose in a save dialog. The name is the default, not fixed | CSV, no index column |
| `batch_results.xlsx` | **Run Batch**, panel 5 | The output directory | XLSX via `openpyxl`, no index column |
| `<stem>_upsampled.tif` | **Run Batch**, one per input | The output directory | TIFF, see the warning below |
| `<stem>_mask.tif` | **Run Batch**, one per input | The output directory | TIFF, integer instance labels |

`<stem>` is the input filename without its extension, so `donor4_scan2.jpk-qi-image` gives
`donor4_scan2_upsampled.tif` and `donor4_scan2_mask.tif`.

!!! warning "Re-running a batch overwrites"

    Files are written with fixed names into the output directory. A second run into the same
    folder replaces `batch_results.xlsx` and every TIFF whose input file has the same stem. Use a
    fresh output directory per run, or per parameter set.

## Single-image CSV

One row per fenestration, in ascending label order. Columns in this order:

| # | Column | Unit | Meaning |
|---|---|---|---|
| 1 | `Label` | none | The integer label Cellpose assigned to this pore. Matches the value in the mask image. |
| 2 | `Area_nm2` | nm² | Pixel count of the mask, converted with the output pixel area. |
| 3 | `Perimeter_nm` | nm | Length of the digitized mask boundary. |
| 4 | `Equivalent_Diameter_nm` | nm | Diameter of a circle with the same area as the mask. |
| 5 | `Equivalent_Diameter_Upsampled_Pixels` | pixels | The same quantity on the upsampled grid, before unit conversion. |
| 6 | `Equivalent_Diameter_Raw_Pixels` | pixels | The same quantity divided by the upsampling factor, so it is expressed on the original scan's grid. |
| 7 | `Eccentricity` | none, 0 to 1 | Eccentricity of the ellipse with the same second moments as the mask. |

Definitions and the arithmetic behind columns 2 to 7 are on [Metrics & units](metrics.md).

!!! note "Porosity is not in the single-image CSV"

    It is calculated, but only shown in the message box that appears after saving, as a
    percentage. If you need it in a file, note it down at that point or use a batch run of one
    image.

The single-image path writes no images. If you want the upsampled scan or the mask as files, use
[Batch analysis](../guide/step5-batch.md).

## Batch workbook

`batch_results.xlsx` has one row per fenestration across all images, in the order the files were
processed. It is the seven columns above with one prepended and one appended. This order holds
only when the first image processed has at least one detected pore; see the warning below.
Address columns by name, never by position:

| # | Column | Unit | Meaning |
|---|---|---|---|
| 1 | `Image_Name` | none | The input file stem, without extension. |
| 2 | `Label` | none | Pore label within that image. Labels restart at 1 for each image, so `Image_Name` plus `Label` is the unique key. |
| 3 | `Area_nm2` | nm² | As above. |
| 4 | `Perimeter_nm` | nm | As above. |
| 5 | `Equivalent_Diameter_nm` | nm | As above. |
| 6 | `Equivalent_Diameter_Upsampled_Pixels` | pixels | As above. |
| 7 | `Equivalent_Diameter_Raw_Pixels` | pixels | As above. |
| 8 | `Eccentricity` | none, 0 to 1 | As above. |
| 9 | `Porosity` | fraction, 0 to 1 | Pore area summed over total image area. This is a **per-image** value, written identically on every row belonging to that image. |

Take `Porosity` with `groupby("Image_Name").first()`, never with `sum()` or `mean()` over rows.
Averaging it over rows weights each image by its pore count.

!!! warning "An image with no detected pores contributes no rows"

    Such an image contributes no rows, and the completion dialog still counts it as processed. A
    genuinely pore-free scan and a failed segmentation look identical: both are missing.
    Cross-check the row count in `Image_Name` against the number of files in your input folder.
    See [Known issues](../caveats/known-issues.md).

    It is not entirely absent, though: it still contributes the `Image_Name` and `Porosity`
    columns. Columns are unioned in order of first appearance, so if such an image is processed
    first, the header comes out `Image_Name, Porosity, Label, Area_nm2, ...` with `Porosity` in
    column 2 rather than column 9. Read the workbook by column name, never by column number.

## The mask TIFF

`<stem>_mask.tif` holds **instance labels**, not a binary mask. Background is 0, and each pore has
its own integer value that matches the `Label` column for that image. `mask.max()` is the number
of pores found. To get a binary mask, use `mask > 0`.

The mask is on the upsampled grid, so its dimensions are the upsampling factor times the raw scan.

## The upsampled TIFF

!!! warning "This file has two different meanings"

    What `<stem>_upsampled.tif` contains depends on the method used in panel 2, and for the
    deep-learning routes also on the **Apply Post-DL Sharpening** checkbox:

    | Method | Checkbox | Data type | Values are |
    |---|---|---|---|
    | `CLAHE (CPU)` | not consulted | `uint16` | contrast-equalized and rescaled to the full 0 to 65535 range, with no physical meaning |
    | `HAT` / `SwinIR` | off | `float32` | physical height, in the units the JPK file was read in |
    | `HAT` / `SwinIR` | on | `uint16` | contrast-equalized and rescaled to the full 0 to 65535 range, with no physical meaning |

    The CLAHE route always produces the `uint16` form, whatever the checkbox says, because
    equalization is the last step of that route and the checkbox is not read at all when the
    method is CLAHE. A CLAHE upsampled TIFF carries no height information.

    Nothing in the filename, the workbook, or the TIFF metadata records which one you got. Two
    output folders from two runs are indistinguishable by inspection alone.

The measured columns are derived from the mask, not from these pixel values, so the units of this
file never enter the CSV or the workbook. Height readings taken off the TIFF do depend on it.
Note also that ticking the box changes the pixel values Cellpose then segments, so it can change
the masks as well.

!!! tip

    Record the method and the state of that checkbox in your lab notebook alongside the output
    folder path. You can check an existing file afterwards with `tifffile.imread(path).dtype`:
    `float32` means heights, `uint16` means equalized.

## Reading the outputs in Python

```python
import pandas as pd
import tifffile

# --- single image ---
df = pd.read_csv("fenestration_metrics.csv")
print(len(df), "pores")
print(df["Equivalent_Diameter_nm"].describe())

# --- batch ---
batch = pd.read_excel("results/batch_results.xlsx", engine="openpyxl")

per_image = batch.groupby("Image_Name").agg(
    n_pores=("Label", "count"),
    mean_diameter_nm=("Equivalent_Diameter_nm", "mean"),
    total_pore_area_nm2=("Area_nm2", "sum"),
    porosity=("Porosity", "first"),   # per-image value, do not average over rows
)
print(per_image)

# --- images ---
mask = tifffile.imread("results/donor4_scan2_mask.tif")
print(mask.max(), "labeled pores; background is 0")

up = tifffile.imread("results/donor4_scan2_upsampled.tif")
print(up.dtype)   # float32 = physical height, uint16 = equalized and rescaled
```

Isolating one pore from the mask, using its `Label` value:

```python
pore_7 = mask == 7
print(pore_7.sum(), "pixels")
```
