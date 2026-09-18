# The scale-domain question

This page explains the one condition your scans have to meet for a FenestRA deep-learning result to mean anything: the pixel scale you acquired at. Nothing in the plugin checks it for you.

## What the models were trained on

The HAT and SwinIR checkpoints were trained to invert a **synthetic ×4 degradation**. High-resolution AFM scans were downsampled by 4, and the network learned to recover the original from the downsampled copy.

The high-resolution side of that training set has a **median scale of 25.00 nm/px**, with a range of 15.62 to 42.86 nm/px across n = 130 scans. The low-resolution side, which is what the network actually receives as input, is four times coarser: **median about 100 nm/px**, range about 62.5 to 171 nm/px.

So the network's input domain is roughly 60 to 170 nm/px, centered on 100.

## What FenestRA does with your scan

`process_jpk` reads the scan and its scale together (`pipeline.py:57-62`). The scale is stored on the widget and displayed in panel 1. It is then not used again until the very end, where it converts pixels to nanometers (`_widget.py:542`, `pipeline.py:509`).

The array itself goes straight into the ×4 network with no scale check anywhere in between. If you load a scan acquired at 25 nm/px, the network is asked to upsample something four times finer than anything it saw in training, and the metrics come back on a 6.25 nm/px grid.

!!! warning
    Nothing warns and nothing fails. An out-of-domain scan produces a complete, plausible-looking super-resolved AFM image and a full table of diameters. The only signal available to you is the `Scale` value printed in panel 1.

## Where your scan sits

The output grid is always `input scale ÷ 4` for the deep-learning methods (`pipeline.py:509`).

| Input scale (the `Scale` in panel 1) | Output grid after ×4 | Where that sits |
|---|---|---|
| 200 nm/px | 50.0 nm/px | Coarser than any training scan. Out of domain. |
| 171 nm/px | 42.9 nm/px | Coarse edge of the training range. |
| **100 nm/px** | **25.0 nm/px** | **The median training case. This is the target.** |
| 62.5 nm/px | 15.6 nm/px | Fine edge of the training range. |
| 50 nm/px | 12.5 nm/px | Finer than any training scan. Out of domain. |
| 25 nm/px | 6.25 nm/px | Four times finer than the median training input, 2.5× finer than the finest. Well out of domain. |

The conversion arithmetic itself is correct. At 100 nm/px input and factor 4, a pore measuring 8 pixels across in the output is reported as 200 nm.

## What to do

**When acquiring.** Set the scan so the pixel size lands near 100 nm/px. That puts the ×4 output near the 25 nm/px training domain, which is the condition the checkpoints were fitted under.

**Before trusting a result.** Read the `Scale` line in panel 1 after loading. If it is well outside roughly 60 to 170 nm/px, you are outside the domain.

**When writing it up.** Report the acquired pixel scale for every scan alongside the reported diameters. If a scan sits outside the band, say so, and treat its numbers as exploratory rather than quantitative. Do not pool in-domain and out-of-domain scans into one distribution.

!!! tip
    A scan acquired at about 25 nm/px is already at the resolution the network was trained to *produce*. If you have such a scan, CLAHE at factor 1 is a more defensible route than asking the ×4 network for another factor of four. Note that **Run Cellpose** always operates on the upsampled array (`_widget.py:471`) and refuses to start until an upsampling step has run (`_widget.py:459-461`), so CLAHE at factor 1 is the only way to segment at the acquired resolution from inside the plugin.

??? note "Why this is a scientific limit and not a missing feature"
    The degradation used in training is synthetic. The network learned to invert one specific downsampling operator, not the physical resolution limit of the AFM tip. That makes the whole approach circular unless the input matches the degradation domain it was fitted on. A dialog box warning the user does not remove the circularity, which is why the guidance here is an acquisition protocol rather than a code change.

## The normalization mismatch

The transform applied to your image before inference is not the transform the model was trained under. This is a known limitation of the current release, stated here rather than worked around.

**In training:** a percentile normalization that deliberately does **not** clip. Under that transform, all 130 training scans exceeded 1.0 at the top, and 124 of 130 went below 0.0 at the bottom. The network was fitted to produce values outside [0, 1] and it learned to do so.

**In the plugin:**

1. Per-image **min-max** rescaling to exactly [0, 1] using that single image's own minimum and maximum (`inference.py:98-106`).
2. Inference.
3. `np.clip(out, 0, 1)` before anything else (`inference.py:138`).
4. Restore physical height with `out * (vmax - vmin) + vmin` (`inference.py:141`).

Two consequences follow.

**The clip truncates the tails the network was trained to produce.** Step 3 removes exactly the range that 130 of 130 training scans occupied above 1.0 and 124 of 130 occupied below 0.0. Those tails are the extremes of the height map: the deepest points inside a pore and the highest points on the membrane. The clipped array is what Cellpose segments and what `regionprops` measures, so the truncation lands directly on the pore rim, which is precisely where the boundary between pore and membrane is decided.

**One bright speck rescales the whole scan.** Step 1 uses the image minimum and maximum, so a single contamination particle or a scan artifact sets `vmax` and compresses the rest of the height range toward zero before the network ever sees it.

!!! warning
    Diameters and porosity produced by this pipeline carry a systematic bias at the pore rim. It is a bias, not noise, so it does not average out across a batch. Treat comparisons between conditions processed identically as more reliable than absolute pore sizes.

## Tiling and window artifacts

Inference runs tiled at 256 pixels with a 32-pixel overlap (`pipeline.py:100`, `inference.py:146-150`). Overlapping regions are combined by accumulating whole tiles and dividing by a coverage count (`inference.py:181-184`). That is a boxcar average over the overlap, not a feathered blend, so a hard transition can remain visible at tile edges.

Separately, HAT partitions the image into 16-pixel windows (`inference.py:48`), which is known to leave a periodic fingerprint in the output.

!!! warning
    Fine periodic texture in a FenestRA output has at least two possible non-biological sources. Do not interpret regular sub-100 nm patterning in the membrane as structure without independent evidence.

## Related pages

- [Known issues](known-issues.md) for the specific defects and their workarounds.
- [Metrics](../reference/metrics.md) for how a pixel becomes a nanometer.
- [Troubleshooting](troubleshooting.md) if you have an error message rather than a suspect number.
