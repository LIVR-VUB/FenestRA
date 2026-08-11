# Metrics & units

How a pixel becomes a nanometer, and exactly what each measured quantity is. Read this before
quoting a diameter or a porosity in a figure legend.

## From pixels to nanometers

The JPK file carries its own scale, in nanometers per pixel. Upsampling by a factor of `f`
subdivides each original pixel into `f × f` smaller ones, so the output grid is `f` times finer.
Every measurement is made on the output grid and converted with that finer value.

```python
upsampled_scale_nm = pixel_to_nm / upsample_factor

Area_nm2                       = region.area                * upsampled_scale_nm ** 2
Perimeter_nm                   = region.perimeter           * upsampled_scale_nm
Equivalent_Diameter_nm         = region.equivalent_diameter * upsampled_scale_nm
Equivalent_Diameter_Upsampled_Pixels = region.equivalent_diameter
Equivalent_Diameter_Raw_Pixels = region.equivalent_diameter / upsample_factor
Eccentricity                   = region.eccentricity

porosity = sum(Area_nm2) / (masks.size * upsampled_scale_nm ** 2)
```

`upsample_factor` is 4 for HAT and SwinIR, always. For CLAHE it is the **Factor** spinbox value.

### Worked example

A scan acquired at 100 nm/px, upsampled ×4:

```text
pixel_to_nm         = 100 nm/px       (read from the JPK file)
upsample_factor     = 4
upsampled_scale_nm  = 100 / 4 = 25 nm per output pixel

A pore 8 output pixels across:
  equivalent diameter = 8 px  x  25 nm/px  =  200 nm

A pore covering 50 output pixels:
  area = 50 px  x  (25 nm)^2  =  31 250 nm^2
```

The direction is right: upsampling makes each pixel represent *less* physical distance, so the
number of nanometers per pixel goes down.

!!! warning "The factor is read when you click, not when you run"

    **Quantify Fenestrations** takes the upsampling factor from the **Method** dropdown as it
    stands at that moment. If you run CLAHE at Factor 2 and then switch the dropdown to HAT before
    clicking Quantify, the plugin assumes ×4 and every length comes out half its true value.
    Do not touch the Method dropdown between running an upsampling and quantifying. See
    [Known issues](../caveats/known-issues.md).

## The metrics

All of them come from `regionprops` in scikit-image, applied to the Cellpose label image.

### Area

`region.area` is the number of pixels in the mask. Multiplying by the output pixel area gives
`Area_nm2`. This is a projected, top-down area. It says nothing about pore depth.

### Perimeter

`Perimeter_nm` is the length of the mask boundary as measured on a pixel grid, not the length of a
smooth contour. scikit-image estimates it from the arrangement of boundary pixels, weighting
straight and diagonal steps differently.

Two consequences worth knowing:

- Perimeter is more sensitive to segmentation roughness than area is. A ragged mask edge adds
  boundary length while barely changing the pixel count, so a noisy segmentation inflates
  perimeter and leaves area almost intact.
- Any shape index built from both, such as circularity `4πA / P²`, inherits that sensitivity and
  is biased low for rough masks. Compare such indices only between runs with identical
  segmentation settings.

### Equivalent diameter

`region.equivalent_diameter` is the diameter of a circle whose area equals the mask area:

```text
equivalent_diameter = sqrt(4 * area / pi)
```

It is a size summary derived from area, not a measured width. For a pore that is not round it
does not correspond to any particular chord across the shape. An elongated slit and a round pore
of the same area get the same equivalent diameter, which is why `Eccentricity` is reported
alongside it.

Three columns carry the same quantity in different units: `Equivalent_Diameter_nm`,
`Equivalent_Diameter_Upsampled_Pixels` (on the output grid), and `Equivalent_Diameter_Raw_Pixels`
(divided by the upsampling factor, so on the original scan's grid). The raw-pixel column is useful
for judging whether a pore was resolved at acquisition: a value near or below 1 means the pore was
smaller than a single acquired pixel and the network invented its shape.

### Eccentricity

`region.eccentricity` is the eccentricity of the ellipse that has the same second central moments
as the mask. It is dimensionless and takes no unit conversion:

| Value | Shape |
|---|---|
| 0 | a circle |
| 0.5 | a moderately elongated ellipse, roughly 1.15:1 |
| 0.9 | strongly elongated, roughly 2.3:1 |
| 1 | a line segment |

Because it is a shape descriptor, it is unaffected by the pixel scale, but it is affected by
segmentation quality: a mask that merges two adjacent pores reads as highly eccentric.

### Porosity

Porosity is the summed pore area divided by the area of the whole image, expressed as a fraction
between 0 and 1. The message box after **Quantify Fenestrations** shows it as a percentage; the
`Porosity` column in `batch_results.xlsx` stores the fraction.

```text
porosity = sum of all Area_nm2  /  (number of image pixels x pixel area)
```

The denominator is the entire image, including any region that is not cell: substrate, holes, the
edge of the cell, and anything else in frame. Porosity therefore depends on how the scan was
framed as much as on the cell. Two scans of the same cell with different amounts of background
give different porosities.

!!! tip

    If your scans do not all have the same proportion of cell in frame, porosity is not comparable
    between them. Report pore density and the diameter distribution as well, and state how the
    field of view was chosen.

Porosity is computed once per image. In the batch workbook it is written identically onto every
row of that image, so take it with `groupby("Image_Name").first()` rather than averaging over
rows. See [Output files](outputs.md).

## What the size filter removes

Cellpose is called with `min_size=15`, a fixed threshold in pixels of mask area. Any pore smaller
than 15 output pixels is discarded before it reaches the table. The threshold is in pixels, so
what it means in nanometers changes with your acquisition scale.

An area of 15 pixels corresponds to an equivalent diameter of `sqrt(4 × 15 / π) ≈ 4.37` output
pixels:

| Acquisition scale | Output grid (÷4) | Smallest pore that survives |
|---|---|---|
| 200 nm/px | 50 nm/px | ~219 nm |
| 100 nm/px | 25 nm/px | ~109 nm |
| 50 nm/px | 12.5 nm/px | ~55 nm |
| 25 nm/px | 6.25 nm/px | ~27 nm |
| 12.5 nm/px | 3.125 nm/px | ~14 nm |

LSEC fenestrations are commonly reported in the region of 50 to 200 nm across, so at coarse
acquisition scales this filter removes a real part of the distribution, silently and without
appearing anywhere in the output. It also truncates the lower tail of every diameter histogram you
produce, which biases the mean upward.

There is no control for it in the interface. If it matters to your measurement, state the value
and the resulting cutoff in your methods.

!!! note "The scale question sits underneath all of this"

    The arithmetic on this page is correct for any input scale. Whether the *image* the
    measurements were made on is trustworthy is a separate question, and it depends on how close
    your acquisition scale is to the one the models were trained for. See
    [The scale-domain question](../caveats/scale-domain.md).
