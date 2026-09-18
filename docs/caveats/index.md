# Caveats & Limits

This section records the situations in which FenestRA produces a number you should not publish. Read it before you report fenestration diameters, porosity, or counts.

The pipeline rarely stops with an error. It will load a scan, upsample it, segment it, and write a CSV even when the input is far outside the domain the network was trained on, even when the Cellpose model actually running is not the one the interface names, and even when a scan contributed no rows at all to the batch workbook. The failure to watch for here is a plausible number, not a crash.

## Ranked by how likely they are to reach a figure

| Caveat | What it does to your measurement | Detail |
|---|---|---|
| The scan's pixel scale is never checked against the training domain | A scan acquired far from about 100 nm/px is upsampled out of domain. The image looks like an AFM image and the numbers are not supported by anything the model was trained on. | [The scale-domain question](scale-domain.md) |
| An empty **CP Model** field runs `cpsam`, and **Diameter** is a rescale factor, not a size | Your segmentation may come from a different network, at a different working scale, than the labels describe. | [Known issues](known-issues.md#6-cellpose-4-label-drift) |
| The network output is clipped to [0, 1] before physical height is restored | The clip removes exactly the height tails the network was trained to produce, and the clipped array is what gets segmented. The bias lands at the pore rim, where the boundary is decided. | [The scale-domain question](scale-domain.md#the-normalization-mismatch) |
| `min_size=15` px is hardcoded in both Cellpose calls | Small pores are removed before you ever see them. The cutoff in nanometers moves with your output grid. | [Known issues](known-issues.md#hardcoded-settings-that-move-your-numbers) |
| Cellpose contrast normalization is recomputed per image | In a batch run every image gets its own stretch, so rows in one `batch_results.xlsx` are not strictly comparable to each other. | [Known issues](known-issues.md#hardcoded-settings-that-move-your-numbers) |
| Images where Cellpose finds nothing contribute no rows to the batch workbook | Your *n* is smaller than the completion dialog reports, and a genuinely empty scan looks identical to a failed segmentation. | [Known issues](known-issues.md#1-batch-drops-images-where-cellpose-finds-nothing) |
| **Quantify Fenestrations** reads the Method dropdown at the moment you click it | Change the dropdown after upsampling and every diameter is converted with the wrong factor, with no warning. | [Known issues](known-issues.md#2-quantify-reads-the-live-method-dropdown) |
| `<stem>_upsampled.tif` means two different things depending on one checkbox | Heights measured off that file are in physical units or in normalized 16-bit counts, and nothing in the file records which. | [Known issues](known-issues.md#hardcoded-settings-that-move-your-numbers) |
| Tiles are blended with a boxcar average, and HAT's window partition leaves a periodic fingerprint | Fine periodic texture in an output image can come from the reconstruction rather than from the membrane. | [The scale-domain question](scale-domain.md#tiling-and-window-artifacts) |
| The displayed layer scale assumes ×4 | In the 4-pane grid, Raw and Upsampled look aligned at a common physical size even when the CLAHE factor was not 4. Exported numbers are unaffected. | [Known issues](known-issues.md#3-the-hardcoded-025-layer-scale) |

If you are chasing an error message rather than a suspect number, go to [Troubleshooting](troubleshooting.md).

## Before you publish

Work through this list once per dataset. Each item takes under a minute and each one corresponds to a caveat above.

1. **Record the Scale of every scan.** Panel 1 prints `Scale: <x> nm/px` after loading (`_widget.py:340`). For the ×4 deep-learning models an input near 100 nm/px puts the output near the 25 nm/px training domain. Put the value in your methods section, not only in your notebook.
2. **Look at the masks over the image before exporting.** Use **Arrange 4-Pane Grid** in panel 4. Judge the boundaries against the upsampled image, and remember that the physical alignment of the panes is only correct at factor 4.
3. **Do not touch the Method dropdown between Run Upsampling and Quantify Fenestrations.** If you did, set it back to the method you actually ran and quantify again.
4. **Check the image count in `batch_results.xlsx`.** Count distinct `Image_Name` values and compare with the number of `_mask.tif` files in the output folder. A mask file with no matching rows is a scan where Cellpose found nothing.
5. **Record whether Apply Post-DL Sharpening was ticked.** With it off, `<stem>_upsampled.tif` is float32 in physical height units. With it on, it is uint16 rescaled to the full range.
6. **Compare the smallest pore you claim against the `min_size` floor.** On a 6.25 nm/px grid the 15-pixel floor removes anything below roughly 27 nm equivalent diameter. On a 25 nm/px grid, roughly 110 nm.
7. **Name the Cellpose model that actually ran.** If you left CP Model empty, that is `cpsam` under Cellpose 4, not cyto2.
8. **Do not read fine periodic texture as biology.** Two non-biological sources are known, and neither is labeled in the output.
9. **Keep the checkpoint filename and the architecture with the results.** Neither the CSV nor the workbook records them. From 0.3.0 `fenestra.__version__` does report the installed release correctly and is safe to quote; on 0.2.11 and earlier it always said `0.0.1` (see [Known issues](known-issues.md#11-the-package-reports-version-001)).

!!! note
    The model weights are not publicly available yet. They are released with the peer-reviewed manuscript, so a reader cannot currently reproduce a deep-learning result from the public repository alone. Say so in your methods section, and cite the checkpoint by name.
