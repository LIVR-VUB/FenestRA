# Known issues

Verified defects in FenestRA 0.2.11, ordered by how much they can change a published number rather than by how hard they are to fix. Each entry gives what you observe, why it happens, what it does to your data, and a workaround where one exists.

## Severity

| Label | Meaning |
|---|---|
| **Silent** | You get a wrong or incomplete number with no error. |
| **Loud** | The run stops with an error message. Your data is not affected. |
| **Nuisance** | Affects how you work, not what you measure. |
| **Cosmetic** | Affects reported metadata only. |

| # | Issue | Severity |
|---|---|---|
| 1 | [Batch drops images where Cellpose finds nothing](#1-batch-drops-images-where-cellpose-finds-nothing) | Silent |
| 2 | [Quantify reads the live Method dropdown](#2-quantify-reads-the-live-method-dropdown) | Silent |
| 3 | [The hardcoded 0.25 layer scale](#3-the-hardcoded-025-layer-scale) | Silent (display) |
| 4 | [Stale Docker images carry the old ENTRYPOINT](#4-stale-docker-images-carry-the-old-entrypoint) | Loud |
| 5 | [No scale check before inference](#5-no-scale-check-before-inference) | Silent |
| 6 | [Cellpose 4 label drift](#6-cellpose-4-label-drift) | Silent |
| 7 | [Only embed_dim 180 checkpoints load](#7-only-embed_dim-180-checkpoints-load) | Loud |
| 8 | [SwinIR tiling fails on certain image heights](#8-swinir-tiling-fails-on-certain-image-heights) | Loud |
| 9 | [CLAHE freezes the GUI thread](#9-clahe-freezes-the-gui-thread) | Nuisance |
| 10 | [Developer paths are pre-filled in the UI](#10-developer-paths-are-pre-filled-in-the-ui) | Nuisance |
| 11 | [The package reports version 0.0.1](#11-the-package-reports-version-001) | Cosmetic |

---

### 1. Batch drops images where Cellpose finds nothing

**Severity:** Silent

**What you observe.** The batch run completes and the dialog reports `Successfully processed 24 images.` The workbook `batch_results.xlsx` contains fewer than 24 distinct `Image_Name` values.

**Why.** When `regionprops` finds no objects, `quantify_fenestrations` returns an empty `DataFrame` (`pipeline.py:199`). The batch loop then adds columns to it with `df.insert(0, "Image_Name", base_name)` and `df["Porosity"] = porosity` (`pipeline.py:431-432`), which gives the frame columns but still zero rows. `pd.concat` at `pipeline.py:437` therefore contributes nothing for that image. Verified on pandas 2.2.2.

**Consequence for your data.** Your *n* is smaller than the number in the completion dialog, and the shortfall is not reported anywhere. A scan with genuinely zero fenestrations and a scan whose segmentation failed are both absent, and are indistinguishable in the workbook.

**Workaround.** `<stem>_mask.tif` is written for every image (`pipeline.py:426`) before quantification happens, so the mask files are a complete record of what was processed. Count them against the distinct `Image_Name` values:

```bash
ls "$OUTDIR"/*_mask.tif | wc -l
```

Any mask file with no rows in the workbook is a zero-detection image. Open it in napari to decide whether the scan was empty or the segmentation failed.

---

### 2. Quantify reads the live Method dropdown

**Severity:** Silent

**What you observe.** Nothing. The CSV saves normally and the porosity dialog looks reasonable.

**Why.** `on_quantify` recomputes the upsampling factor from the current state of the interface rather than from what was run (`_widget.py:483-484`):

```python
method = self.combo_method.currentText()
factor = self.spin_up_factor.value() if "CLAHE" in method else 4.0
```

The widget never records which method produced `self.upsampled_image`.

**Consequence for your data.** Run CLAHE at factor 2, switch the Method dropdown to HAT, then click **Quantify Fenestrations**. The scale becomes `pixel_to_nm / 4` instead of `pixel_to_nm / 2`. The length metrics `Perimeter_nm`, `Equivalent_Diameter_nm` and `Equivalent_Diameter_Raw_Pixels` are then reported at half their correct value, and `Area_nm2`, which scales with the square of the pixel size (`pipeline.py:191`), at a quarter. There is no warning.

**Workaround.** Do not change the Method dropdown between clicking **Run Upsampling** and clicking **Quantify Fenestrations**. If you already did, set the dropdown back to the method you actually ran and quantify again. The mask in the viewer is unaffected, so nothing needs to be recomputed.

---

### 3. The hardcoded 0.25 layer scale

**Severity:** Silent, display only

**What you observe.** In the 4-pane grid, `Raw AFM` and `Upsampled AFM` appear at the same physical size and look aligned.

**Why.** The layer scale is fixed at `scale=(0.25, 0.25)` in three places, with the comment `# assumed 4x upsampling`: the upsampled image (`_widget.py:399`), the mask (`_widget.py:431`) and the RGB overlay (`_widget.py:466`). The Factor spinbox accepts 1 to 10.

**Consequence for your data.** At any CLAHE factor other than 4 the panes are displayed at mismatched physical scales while looking correct. Exported numbers are **not** affected: `on_quantify` derives the scale from the spinbox value, not from the layer.

**Workaround.** Use factor 4 for CLAHE if you rely on the grid view for judgment. Otherwise correct the layer in the napari console:

```python
f = 2  # the CLAHE factor you actually used
for name in ("Upsampled AFM", "Cellpose Masks", "Overlay"):
    if name in viewer.layers:
        viewer.layers[name].scale = (1 / f, 1 / f)
```

---

### 4. Stale Docker images carry the old ENTRYPOINT

**Severity:** Loud

**What you observe.** A **DL Error** dialog reading `Background thread error: Container DL Inference failed:` followed by:

```text
python: can't open file '/opt/python': [Errno 2] No such file or directory
```

**Why.** `containers/Dockerfile` used to end with `ENTRYPOINT ["python"]`, while the plugin appends its own `python /opt/dl_project/scripts/inference.py ...` as the command (`pipeline.py:91` interactive, `pipeline.py:249` batch). Docker concatenates ENTRYPOINT and CMD, so the process inside the container was `python python /opt/dl_project/scripts/inference.py`, and Python treated the literal string `python` as the script path.

**Consequence for your data.** None. Nothing runs.

**Fix.** The `ENTRYPOINT` line has been removed from the Dockerfile, so an image built from the current repository is correct. Seeing this error means your image was built from an older checkout: `git pull`, then rebuild.

```bash
cd containers
docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
```

Apptainer/Singularity was never affected, because `singularity exec` bypasses `%runscript` and genuinely needs the explicit `python`. That asymmetry is why only one of the two recipes had to change.

!!! info
    Both the original diagnosis and the fix are reasoned from documented ENTRYPOINT/CMD semantics plus the two source files. The image is not built on the machine where the documentation was verified, so neither the failure nor the fix was executed.

---

### 5. No scale check before inference

**Severity:** Silent

**What you observe.** Nothing. The output image looks like a plausible AFM scan.

**Why.** `pixel_to_nm` is read at load time (`pipeline.py:24-29`), displayed in panel 1, and then not consulted again until it converts pixels to nanometers at the end (`_widget.py:486`, `pipeline.py:429`). No code compares it with the domain the checkpoints were trained on.

**Consequence for your data.** A scan acquired far from about 100 nm/px is upsampled outside the training domain and yields metrics with no support behind them.

**Workaround.** Check the `Scale` line in panel 1 yourself before running. Full explanation and a table of scales in [The scale-domain question](scale-domain.md).

---

### 6. Cellpose 4 label drift

**Severity:** Silent

**What you observe.** The CP Model field shows the placeholder `Leave empty for cyto2` and the diameter row is labeled `Diameter (0=auto)`. Both describe Cellpose 2 behavior. Verified against cellpose 4.1.1, the version installed in `fenestra-env`.

**Why.**

| Control | What the code does | What Cellpose 4 does |
|---|---|---|
| CP Model left empty | `models.CellposeModel(gpu=..., model_type="cyto2")` (`pipeline.py:134`, `:290`) | `models.py:108-110` logs that `model_type` is not used in v4.0.1+ and ignores it. You get `cpsam`. |
| Diameter | passed straight to `eval()` | `models.py:272-273` computes `image_scaling = 30. / diameter`. The default 30 gives scaling 1.0, a no-op. `0` fails the `> 0` test and is also a no-op. |

**Consequence for your data.** Leaving CP Model empty runs the Cellpose-SAM default model, not cyto2, so a methods section that says "cyto2" is wrong. Diameter 0 and diameter 30 are the same setting, and there is no automatic diameter estimation in Cellpose 4. Values below 30 upscale the image before segmentation and values above 30 downscale it.

**Workaround.** Point CP Model at an explicit checkpoint, then confirm it actually loaded. The fallback triggers on any path that does not exist, not only on an empty field (`pipeline.py:126`, `:283`), and unlike the DL model field (checked at `_widget.py:325`) the CP Model path is never validated (`_widget.py:410`). A typo or a moved file therefore runs `cpsam` with no dialog. Check the terminal for the cellpose warning that `model_type` is not used in v4.0.1+, which appears only on the fallback branch. If you do leave the field empty, record `cpsam` in your methods. Read Diameter as a rescale factor of `30 / value`, not as an object size in pixels. See [Cellpose Segmentation](../guide/step3-segmentation.md).

---

### 7. Only embed_dim 180 checkpoints load

**Severity:** Loud

**What you observe.** `Background thread error: Container DL Inference failed:` followed by a `RuntimeError` from `load_state_dict` listing size mismatches or missing and unexpected keys. On the batch path the same message appears without the `Background thread error:` prefix (`pipeline.py:261`).

**Why.** The architecture is hardcoded, not derived from the checkpoint. Both branches of `build_model` fix `embed_dim=180`, `depths=[6, 6, 6, 6, 6, 6]` and `num_heads=[6, 6, 6, 6, 6, 6]` (`inference.py:47-53` for HAT, `:65-70` for SwinIR), and weights are loaded with `strict=True` (`inference.py:87`). Checkpoints trained with `embed_dim=96` and `depths=[6]*4`, the smaller variants, do not match.

**Consequence for your data.** None. The run stops before producing anything. The smaller model variants cannot currently be used from the plugin.

**Workaround.** Use a checkpoint built with `embed_dim=180` and `depths=[6]*6`.

!!! danger
    Do not change `strict=True` to `strict=False` to make a mismatched checkpoint load. That converts a loud failure into a silent one: the model would run with partially random weights and still produce a plausible-looking height map.

---

### 8. SwinIR tiling fails on certain image heights

**Severity:** Loud

**What you observe.** During a SwinIR run:

```text
RuntimeError: Padding size should be less than the corresponding input dimension
```

**Why.** The full image is first padded to a multiple of the model's window size, 16 for HAT and 8 for SwinIR (`inference.py:114-118`). Each individual tile is then padded to a multiple of **16** regardless of architecture (`inference.py:166-167`). With `--tile_size 256` and a 32-pixel overlap the stride is 224, so a padded height of 224k + 8 leaves a final tile 8 rows tall that needs 8 rows of reflect padding, and torch requires the padding to be smaller than the dimension being padded. Width goes through the same rule at `inference.py:167`.

The rule applies to the height **after** the whole-image pad, so every raw height from 224k + 1 to 224k + 8 rounds up to the same failing value. Failures therefore come in runs of eight rather than as isolated sizes.

Verified against torch 2.4.0: of the raw heights from 8 to 4096, 2041 need a tile pad under SwinIR and **145 fail**. The failing runs are 225–232, 449–456, 673–680, 897–904, and so on. HAT never needs the tile pad at all, in 0 of 4081 heights tested.

**Consequence for your data.** None. It fails before writing anything, never silently.

**Workaround.** Use HAT, or crop the scan to a safe size. The common square sizes 256, 512, 1024 and 2048 are all safe for SwinIR.

---

### 9. CLAHE freezes the GUI thread

**Severity:** Nuisance

**What you observe.** After clicking **Run Upsampling** with `CLAHE (CPU)` selected, napari stops repainting. The window manager may report it as not responding.

**Why.** `upsample_clahe` is called directly on the Qt thread (`_widget.py:312`), while the deep-learning and Cellpose steps go through `@thread_worker` and stay responsive. The button text is set to `Upsampling in progress...` before the call (`_widget.py:302`) but the window may not repaint to show it.

**Consequence for your data.** None. It is not crashed. The cubic zoom at `pipeline.py:52` is the slow part, and it grows with the square of the factor.

**Workaround.** Wait. The window returns when the `Upsampled AFM` layer appears and the button reads `Run Upsampling` again. Test with a small factor first to gauge the time on your machine.

---

### 10. Developer paths are pre-filled in the UI

**Severity:** Nuisance

**What you observe.** The **DL Model** and **Singularity (.sif)** fields arrive pre-filled with `/home/arka/Desktop/AFM-Project/DL_Upsampling/...`. Switching **Engine** to Docker and back to Singularity replaces whatever you typed in the container field with that same path.

**Why.** The paths are the initial text of the two line edits (`_widget.py:102`, `:125`), and `on_engine_changed` rewrites the container field every time Singularity is selected (`_widget.py:270`).

**Consequence for your data.** None. The plugin refuses to run with a missing model file (`_widget.py:325`) or a missing `.sif` (`_widget.py:331`), so a stale path produces a dialog rather than a wrong result. The Docker tag field is not validated, so a wrong tag surfaces as a docker error instead.

**Workaround.** Choose the Engine first, then set the model and container paths. Re-check both fields after any Engine change.

---

### 11. The package reports version 0.0.1

**Severity:** Cosmetic

**What you observe.**

```text
>>> import fenestra; fenestra.__version__
'0.0.1'
```

while `pip show napari-fenestra` reports `0.2.11`.

**Why.** `src/fenestra/__init__.py:1` holds a static string that was never updated. The packaging metadata in `setup.cfg:3` is the single source of truth and is correct.

**Consequence for your data.** Any provenance you record from the Python attribute, in a bug report or in a script that stamps a version into an output file, will name a version that was never released.

**Workaround.** Take the version from the package metadata:

```bash
pip show napari-fenestra | grep Version
```

---

## Hardcoded settings that move your numbers

These are not defects, but they are fixed in the source, not exposed in the interface, and they change what you measure.

- **`min_size=15` pixels** in both Cellpose calls (`pipeline.py:143`, `:299`). Objects below 15 pixels in area are discarded. That is an equivalent diameter of about 4.4 pixels, which is roughly 27 nm on a 6.25 nm/px output grid and roughly 110 nm on a 25 nm/px grid. The cutoff in nanometers moves with your acquisition scale, so the smallest detectable pore is not constant across a study that mixes scan settings.
- **Per-image contrast normalization.** `normalize={"normalize": True, "percentile": (1.0, 99.0)}` is recomputed for every image (`pipeline.py:139`, `:295`). In a batch run each image gets its own stretch before segmentation, so rows in one `batch_results.xlsx` are not strictly comparable to each other.
- **`<stem>_upsampled.tif` has two possible meanings.** Straight from the container it is float32 in physical height units (`inference.py:143`). With **Apply Post-DL Sharpening** ticked it is uint16 rescaled to the full 16-bit range (`pipeline.py:45`). Nothing in the filename or the workbook records which, so heights measured off those files depend on a checkbox. See [Outputs](../reference/outputs.md).
- **Stale output can be reused in the interactive path.** The interactive deep-learning run writes into one temporary directory that lives for the whole napari session (`_widget.py:26`, `:361`) and takes the first `*.tif*` it finds. `inference.py:101-103` skips a perfectly uniform image and still exits 0, so in that narrow case a previous run's output is loaded instead of an error being raised. Batch is not affected: it creates a fresh temporary directory per image (`pipeline.py:388`).
- **Tile blending is a boxcar average** over the 32-pixel overlap (`inference.py:181-184`), and HAT uses a 16-pixel window partition. Both can leave periodic structure in the output. See [The scale-domain question](scale-domain.md#tiling-and-window-artifacts).

## No automated tests

The repository contains no test suite and no continuous integration configuration. A behavioral change to the pipeline is not caught by anything other than a person looking at the output.

If you upgrade FenestRA, re-run one scan you have already measured and confirm the diameters and porosity match your previous result before processing new data.

## Related pages

- [The scale-domain question](scale-domain.md) for the acquisition condition behind issue 5.
- [Troubleshooting](troubleshooting.md) if you are looking up an error string.
- [Parameters](../reference/parameters.md) for what every control does.
