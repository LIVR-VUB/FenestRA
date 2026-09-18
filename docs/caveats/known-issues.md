# Known issues

!!! danger "Read this first if you have an RTX 50-series GPU"

    **An RTX 5060, 5070, 5080, 5090 or any other Blackwell card cannot run the standard image, and
    the way it fails is designed to waste your time.** Everything installs. The GPU is detected.
    An image built before commit `379b170` reports the card as working; an image built from the
    current repository prints a boxed warning naming your `sm_` level and the build's arch list.
    Then the first deep-learning run dies, hundreds of lines into a traceback, with:

    ```text
    RuntimeError: CUDA error: no kernel image is available for execution on the device
    ```

    Nothing is wrong with your machine, your driver, your Docker install, or your data. The
    bundled PyTorch simply contains no compiled kernels for your GPU's architecture.

    **Fix:** build `containers/Dockerfile.allinone.cu128` instead, and point the launcher at it.
    Full instructions: [All-in-one container](../install/all-in-one.md#first-which-of-the-two-recipes).

    This applies to the native conda + pip install too. Its pinned torch 2.4.0+cu124 stops at
    `sm_90`, and the reference DL container is older still, so on a Blackwell card the `cu128`
    all-in-one image is the only working route.

    Details, and how to tell which image you are running:
    [issue 13](#13-rtx-50-series-blackwell-gpus-cannot-run-the-standard-image).

Verified defects in FenestRA 0.3.0, ordered by how much they can change a published number rather than by how hard they are to fix. Each entry gives what you observe, why it happens, what it does to your data, and a workaround where one exists.

Entries fixed in 0.3.0 are kept rather than deleted, and still describe the old behavior, so that an install of 0.2.11 or earlier remains recognizable from its symptom. Where a label was corrected but the underlying behavior was not, the entry says so explicitly.

## Severity

| Label | Meaning |
|---|---|
| **Silent** | You get a wrong or incomplete number with no error. |
| **Loud** | The run stops with an error message. Your data is not affected. |
| **Nuisance** | Affects how you work, not what you measure. |
| **Cosmetic** | Affects reported metadata only. |
| **Fixed in 0.3.0** | No longer present in the current release. Kept for older installs. |

| # | Issue | Severity |
|---|---|---|
| 1 | [Batch drops images where Cellpose finds nothing](#1-batch-drops-images-where-cellpose-finds-nothing) | Silent |
| 2 | [Quantify reads the live Method dropdown](#2-quantify-reads-the-live-method-dropdown) | Silent |
| 3 | [The hardcoded 0.25 layer scale](#3-the-hardcoded-025-layer-scale) | Silent (display) |
| 4 | [Stale Docker images carry the old ENTRYPOINT](#4-stale-docker-images-carry-the-old-entrypoint) | Loud |
| 5 | [No scale check before inference](#5-no-scale-check-before-inference) | Silent |
| 6 | [Cellpose 4 changed what CP Model and Diameter mean](#6-cellpose-4-label-drift) | Silent — labels corrected in 0.3.0, behavior unchanged |
| 7 | [Only embed_dim 180 checkpoints load](#7-only-embed_dim-180-checkpoints-load) | Loud |
| 8 | [SwinIR tiling fails on certain image heights](#8-swinir-tiling-fails-on-certain-image-heights) | Loud |
| 9 | [CLAHE freezes the GUI thread](#9-clahe-freezes-the-gui-thread) | Nuisance |
| 10 | [Developer paths were pre-filled in the UI](#10-developer-paths-are-pre-filled-in-the-ui) | Fixed in 0.3.0 |
| 11 | [The package reported version 0.0.1](#11-the-package-reports-version-001) | Fixed in 0.3.0 |
| 12 | [The all-in-one image is not the reference DL stack](#12-the-all-in-one-image-is-not-the-reference-dl-stack) | Silent (all-in-one only) |
| **13** | [**RTX 50-series (Blackwell) GPUs: only the `cu128` all-in-one image works**](#13-rtx-50-series-blackwell-gpus-cannot-run-the-standard-image) | **Loud, but only after a long detour** |

---

### 1. Batch drops images where Cellpose finds nothing

**Severity:** Silent

**What you observe.** The batch run completes and the dialog reports `Successfully processed 24 images.` The workbook `batch_results.xlsx` contains fewer than 24 distinct `Image_Name` values.

**Why.** When `regionprops` finds no objects, `quantify_fenestrations` returns an empty `DataFrame` (`pipeline.py:309`). The batch loop then adds columns to it with `df.insert(0, "Image_Name", base_name)` and `df["Porosity"] = porosity` (`pipeline.py:511-512`), which gives the frame columns but still zero rows. `pd.concat` at `pipeline.py:517` therefore contributes nothing for that image. Verified on pandas 2.3.3, the version pinned for the host environment.

**Consequence for your data.** Your *n* is smaller than the number in the completion dialog, and the shortfall is not reported anywhere. A scan with genuinely zero fenestrations and a scan whose segmentation failed are both absent, and are indistinguishable in the workbook.

**Workaround.** `<stem>_mask.tif` is written for every image (`pipeline.py:506`) before quantification happens, so the mask files are a complete record of what was processed. Count them against the distinct `Image_Name` values:

```bash
ls "$OUTDIR"/*_mask.tif | wc -l
```

Any mask file with no rows in the workbook is a zero-detection image. Open it in napari to decide whether the scan was empty or the segmentation failed.

---

### 2. Quantify reads the live Method dropdown

**Severity:** Silent

**What you observe.** Nothing. The CSV saves normally and the porosity dialog looks reasonable.

**Why.** `on_quantify` recomputes the upsampling factor from the current state of the interface rather than from what was run (`_widget.py:539-540`):

```python
method = self.combo_method.currentText()
factor = self.spin_up_factor.value() if "CLAHE" in method else 4.0
```

The widget never records which method produced `self.upsampled_image`.

**Consequence for your data.** Run CLAHE at factor 2, switch the Method dropdown to HAT, then click **Quantify Fenestrations**. The scale becomes `pixel_to_nm / 4` instead of `pixel_to_nm / 2`. The length metrics `Perimeter_nm`, `Equivalent_Diameter_nm` and `Equivalent_Diameter_Raw_Pixels` are then reported at half their correct value, and `Area_nm2`, which scales with the square of the pixel size (`pipeline.py:301`), at a quarter. There is no warning.

**Workaround.** Do not change the Method dropdown between clicking **Run Upsampling** and clicking **Quantify Fenestrations**. If you already did, set the dropdown back to the method you actually ran and quantify again. The mask in the viewer is unaffected, so nothing needs to be recomputed.

---

### 3. The hardcoded 0.25 layer scale

**Severity:** Silent, display only

**What you observe.** In the 4-pane grid, `Raw AFM` and `Upsampled AFM` appear at the same physical size and look aligned.

**Why.** The layer scale is fixed at `scale=(0.25, 0.25)` in three places, with the comment `# assumed 4x upsampling`: the upsampled image (`_widget.py:455`), the mask (`_widget.py:487`) and the RGB overlay (`_widget.py:522`). The Factor spinbox accepts 1 to 10 (`_widget.py:91`).

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

**What you observe.** A **DL Error** dialog reading `Container DL Inference failed:` followed by:

```text
python: can't open file '/opt/python': [Errno 2] No such file or directory
```

**Why.** `containers/Dockerfile` used to end with `ENTRYPOINT ["python"]`, while the plugin appends its own `python /opt/dl_project/scripts/inference.py ...` as the command (`pipeline.py:145-152`). Docker concatenates ENTRYPOINT and CMD, so the process inside the container was `python python /opt/dl_project/scripts/inference.py`, and Python treated the literal string `python` as the script path.

**Consequence for your data.** None. Nothing runs.

On 0.2.11 and earlier the same dialog prefixed the message with `Background thread error:`.

**Fix.** The `ENTRYPOINT` line has been removed from the Dockerfile, so an image built from the current repository is correct. Seeing this error means your image was built from an older checkout: `git pull`, then rebuild.

```bash
cd containers
docker build -t livrvub/dl-upsampling:latest -f Dockerfile ..
```

Apptainer/Singularity was never affected, because `singularity exec` bypasses `%runscript` and genuinely needs the explicit `python`. That asymmetry is why only one of the two recipes had to change.

Since 0.3.0 the command for every engine is built once, by `_build_dl_cmd` (`pipeline.py:109-165`), and run through `_run_dl_inference` (`pipeline.py:168-189`) by both the interactive worker and the batch loop. Before that the argv was copy-pasted into two functions, and a change to one could leave the other behind. The **Local (bundled)** engine launches no container at all (`pipeline.py:114-135`), so it cannot hit this.

!!! info
    Both the original diagnosis and the fix are reasoned from documented ENTRYPOINT/CMD semantics plus the two source files. The image is not built on the machine where the documentation was verified, so neither the failure nor the fix was executed.

---

### 5. No scale check before inference

**Severity:** Silent

**What you observe.** Nothing. The output image looks like a plausible AFM scan.

**Why.** `pixel_to_nm` is read at load time (`pipeline.py:57-62`, shown in panel 1 at `_widget.py:340`), and then not consulted again until it converts pixels to nanometers at the end (`_widget.py:542`, `pipeline.py:509`). No code compares it with the domain the checkpoints were trained on. This is true of all three engines, including **Local (bundled)**.

**Consequence for your data.** A scan acquired far from about 100 nm/px is upsampled outside the training domain and yields metrics with no support behind them.

**Workaround.** Check the `Scale` line in panel 1 yourself before running. Full explanation and a table of scales in [The scale-domain question](scale-domain.md).

---

### 6. Cellpose 4 changed what CP Model and Diameter mean { #6-cellpose-4-label-drift }

**Severity:** Silent. The **labels** were corrected in 0.3.0; the **behavior** they describe has not changed.

**What you observe.** In 0.3.0 the CP Model field shows the placeholder `Leave empty for the Cellpose 4 default (cpsam)` (`_widget.py:193`) and the diameter row is labeled `Diameter (30 = no rescale)` (`_widget.py:208`). Up to and including 0.2.11 the same two controls read `Leave empty for cyto2` and `Diameter (0=auto)`, which described Cellpose 2 behavior and were wrong. Verified against cellpose 4.1.1, the version installed in `fenestra-env`.

**Why.**

| Control | What the code does | What Cellpose 4 does |
|---|---|---|
| CP Model left empty | `models.CellposeModel(gpu=...)` with no `model_type` (`pipeline.py:244`, `:369`) | Loads the built-in default, `cpsam`. Before 0.3.0 the plugin passed `model_type="cyto2"`; `models.py:108-110` logs that `model_type` is not used in v4.0.1+ and ignores it, so that argument never selected cyto2 either. |
| Diameter | passed straight to `eval()` | `models.py:272-273` computes `image_scaling = 30. / diameter`. The default 30 gives scaling 1.0, a no-op. `0` fails the `> 0` test and is also a no-op. |

Because the ignored argument is gone, `setup.cfg:36` now pins `cellpose>=4.0.1` as a hard floor: on cellpose 2 or 3 a `CellposeModel` built without `model_type` has no model to load.

**Consequence for your data.** Unchanged by 0.3.0. Leaving CP Model empty runs the Cellpose-SAM default model, not cyto2, so a methods section that says "cyto2" is wrong. Diameter 0 and diameter 30 are the same setting, and there is no automatic diameter estimation in Cellpose 4. Values below 30 upscale the image before segmentation and values above 30 downscale it.

**Workaround.** Point CP Model at an explicit checkpoint, then confirm it actually loaded. The fallback triggers on any path that does not exist, not only on an empty field (`pipeline.py:224`, `:356`), and unlike the DL model field (checked at `_widget.py:383`) the CP Model path is never validated (`_widget.py:466`). A typo or a moved file therefore runs `cpsam` with no dialog.

!!! warning "The terminal no longer tells you"
    Up to 0.2.11 the fallback branch produced a cellpose log line saying that `model_type` is not used in v4.0.1+, which was the only visible sign that the default model had been substituted. Removing the ignored argument in 0.3.0 also removed that line. Check the CP Model path yourself before the run; nothing prints on the fallback branch now.

If you do leave the field empty, record `cpsam` in your methods. Read Diameter as a rescale factor of `30 / value`, not as an object size in pixels. See [Cellpose Segmentation](../guide/step3-segmentation.md).

---

### 7. Only embed_dim 180 checkpoints load

**Severity:** Loud

**What you observe.** `Container DL Inference failed:` followed by a `RuntimeError` from `load_state_dict` listing size mismatches or missing and unexpected keys. Both the interactive and the batch path raise it from the same place (`pipeline.py:184`) and show identical text. On 0.2.11 and earlier the interactive dialog prefixed this with `Background thread error:`.

**Why.** The architecture is hardcoded, not derived from the checkpoint. Both branches of `build_model` fix `embed_dim=180`, `depths=[6, 6, 6, 6, 6, 6]` and `num_heads=[6, 6, 6, 6, 6, 6]` (`inference.py:47-53` for HAT, `:65-70` for SwinIR), and weights are loaded with `strict=True` (`inference.py:87`). Checkpoints trained with `embed_dim=96` and `depths=[6]*4`, the smaller variants, do not match.

**Consequence for your data.** None. The run stops before producing anything. The smaller model variants cannot currently be used from the plugin, on any engine.

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

**Why.** The full image is first padded to a multiple of the model's window size, 16 for HAT and 8 for SwinIR (`inference.py:114-118`). Each individual tile is then padded to a multiple of **16** regardless of architecture (`inference.py:166-167`). With the hardcoded tile size of 256 (`pipeline.py:100`) and a 32-pixel overlap the stride is 224, so a padded height of 224k + 8 leaves a final tile 8 rows tall that needs 8 rows of reflect padding, and torch requires the padding to be smaller than the dimension being padded. Width goes through the same rule at `inference.py:167`.

The rule applies to the height **after** the whole-image pad, so every raw height from 224k + 1 to 224k + 8 rounds up to the same failing value. Failures therefore come in runs of eight rather than as isolated sizes.

Verified against torch 2.4.0: of the raw heights from 8 to 4096, 2041 need a tile pad under SwinIR and **145 fail**. The failing runs are 225–232, 449–456, 673–680, 897–904, and so on. HAT never needs the tile pad at all, in 0 of 4081 heights tested.

**Consequence for your data.** None. It fails before writing anything, never silently.

**Workaround.** Use HAT, or crop the scan to a safe size. The common square sizes 256, 512, 1024 and 2048 are all safe for SwinIR.

---

### 9. CLAHE freezes the GUI thread

**Severity:** Nuisance

**What you observe.** After clicking **Run Upsampling** with `CLAHE (CPU)` selected, napari stops repainting. The window manager may report it as not responding.

**Why.** `upsample_clahe` is called directly on the Qt thread (`_widget.py:370`), while the deep-learning and Cellpose steps go through `@thread_worker` and stay responsive. The button text is set to `Upsampling in progress...` before the call (`_widget.py:360`) but the window may not repaint to show it.

**Consequence for your data.** None. It is not crashed. The cubic zoom at `pipeline.py:85` is the slow part, and it grows with the square of the factor.

**Workaround.** Wait. The window returns when the `Upsampled AFM` layer appears and the button reads `Run Upsampling` again. Test with a small factor first to gauge the time on your machine.

---

### 10. Developer paths were pre-filled in the UI { #10-developer-paths-are-pre-filled-in-the-ui }

**Severity:** Fixed in 0.3.0. Nuisance up to 0.2.11.

**What you observed up to 0.2.11.** The **DL Model** and **Singularity (.sif)** fields arrived pre-filled with `/home/arka/Desktop/AFM-Project/DL_Upsampling/...`, a path that exists on one developer's machine. Switching **Engine** to Docker and back to Singularity replaced whatever you had typed in the container field with that same path.

**Why.** The paths were the initial text of the two line edits, and `on_engine_changed` rewrote the container field every time Singularity was selected.

**Fix.** Neither path is in the source any more. Both fields now start empty, or at whatever the environment says (`_widget.py:28-32`):

| Variable | Fills |
|---|---|
| `FENESTRA_ENGINE` | which engine is selected at startup |
| `FENESTRA_DL_MODEL` | **DL Model** |
| `FENESTRA_SIF` | the container field under Singularity |
| `FENESTRA_DOCKER_IMAGE` | the container field under Docker, default `livrvub/dl-upsampling:latest` |
| `FENESTRA_CP_MODEL` | **CP Model** |

`on_engine_changed` (`_widget.py:303-329`) now stores the current text under the engine you are leaving and restores the value that engine had, so switching back and forth no longer discards anything you typed. Under **Local (bundled)** the field is disabled and shows the interpreter that will be used, since there is nothing to choose. The [all-in-one image](../install/all-in-one.md) sets these variables in its own Dockerfile, which is why it opens with the Local engine and `/models/best_model_ema.pth` already filled in.

**Consequence for your data.** None, then or now. The plugin refuses to run with a missing model file (`_widget.py:383`) or a missing `.sif` (`_widget.py:389`), so a stale path produced a dialog rather than a wrong result. The Docker tag is still not validated, so a wrong tag surfaces as a docker error instead.

---

### 11. The package reported version 0.0.1 { #11-the-package-reports-version-001 }

**Severity:** Fixed in 0.3.0. Cosmetic up to 0.2.11.

**What you observed up to 0.2.11.**

```text
>>> import fenestra; fenestra.__version__
'0.0.1'
```

while `pip show napari-fenestra` reported `0.2.11`. `src/fenestra/__init__.py` held a static string that was never updated alongside `setup.cfg`.

**Fix.** `src/fenestra/__init__.py:3-10` now reads the version from the installed package metadata with `importlib.metadata.version("napari-fenestra")`, so there is one source of truth and it is `setup.cfg`. The attribute is safe to quote in a methods section:

```text
>>> import fenestra; fenestra.__version__
'0.3.0'
```

In a source tree that was never installed, `PackageNotFoundError` is caught and the attribute reads `0.0.0+unknown`. That string means "running from a checkout", not a release.

**Consequence for your data.** On 0.2.11 and earlier, any provenance recorded from the Python attribute names a version that was never released. If you have old output stamped `0.0.1`, take the real version from your environment records instead:

```bash
pip show napari-fenestra | grep Version
```

---

### 12. The all-in-one image is not the reference DL stack

**Severity:** Silent, and only in the [all-in-one image](../install/all-in-one.md)

**What you observe.** Nothing. Inference runs, the output is a plausible AFM height map, and the metrics look normal.

**Why.** The two backends are built against different versions of PyTorch:

| Backend | Stack |
|---|---|
| `containers/dl_upsampling.def`, `containers/Dockerfile` — the reference | torch 1.14 on `nvcr.io/nvidia/pytorch:23.01-py3`, which is what the method was developed and validated against |
| `containers/Dockerfile.allinone`, the `/opt/venv-dl` environment behind the **Local (bundled)** engine | `torch==2.1.2` with `torchvision==0.16.2` (`containers/Dockerfile.allinone:129`) |

The divergence is deliberate. The torch 1.13 and 1.14 wheels are compiled for sm_37 through sm_86 with no PTX fallback, so they do not start at all on an RTX 40-series card, an H100, or anything newer. Pinning the reference version would have made the all-in-one image unusable on most GPUs bought after 2022. `containers/dl_upsampling.def` is unchanged and still builds the reference stack.

**Consequence for your data.** The architectures, the weights and the arithmetic are identical, but the two stacks are not bit-for-bit equivalent, and nothing in the output records which one produced it. Numbers intended for publication should come from the reference container.

**Workaround.** Use the Singularity or Docker engine against `dl_upsampling.sif` or the `livrvub/dl-upsampling:latest` image for any run you intend to publish, and treat the all-in-one image as the route for getting a working screen in front of a biologist. Record the engine and the backend stack alongside the checkpoint name in your methods.

---

### 13. RTX 50-series (Blackwell) GPUs: only the `cu128` all-in-one image works { #13-rtx-50-series-blackwell-gpus-cannot-run-the-standard-image }

**Severity: loud — but only after a long detour.**

!!! danger "This one cost a real user most of a working day"

    Every individual symptom points somewhere other than the actual cause. The list below is in
    the order the misdirection actually happens, so that anyone hitting it recognises where they
    are and stops looking in the wrong place.

**What you observe.** Everything appears to work. `docker build` succeeds. The container starts.
napari opens in the browser. A `.jpk-qi-image` loads and displays. The container's startup banner
prints your GPU by name. Then **Run Upsampling** fails, and the dialog contains several hundred lines ending in:

```text
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,
so the stacktrace below might be incorrect.
```

Higher up the same output, easy to miss among the deprecation warnings:

```text
NVIDIA GeForce RTX 5070 with CUDA capability sm_120 is not compatible with the
current PyTorch installation. The current PyTorch install supports CUDA
capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

**Why.** PyTorch ships compiled kernels for a fixed list of GPU architectures. The standard image
carries torch 2.4.0+cu124 in the GUI environment and torch 2.1.2 in the deep-learning one; both
were built before Blackwell existed, and stop at `sm_90`. RTX 50-series cards report `sm_120`.

The card is still *visible*: `torch.cuda.is_available()` returns `True`, `torch.cuda.get_device_name()`
returns the right name, and memory can be allocated. Only an actual kernel launch fails — which is
why the error arrives at `F.pad` deep inside `inference.py`, rather than at startup where it would
be obvious.

**What made it expensive.** Four separate things pointed away from the cause:

| Misdirection | Why it looked like something else |
|---|---|
| The banner said `GPU: NVIDIA GeForce RTX 5070 (CUDA 12.4)` | Images built before commit `379b170` only checked `torch.cuda.is_available()`, so they reported a working GPU. Fixed — the banner now compares the card's compute capability against the build's arch list and says plainly when they do not match. |
| The real warning was buried | PyTorch's own `sm_120 is not compatible` warning appears among `UserWarning` lines about deprecated torchvision modules and `torch.meshgrid`, hundreds of lines above the exception. |
| Two images look identical from outside | Once a `cu128` image exists, `docker images` shows both and nothing said which one was running. Fixed — the launcher now prints `Image: livrvub/fenestra:cu128` at startup. |
| `set` does nothing in PowerShell | The documented `set FENESTRA_IMAGE=...` is Command Prompt syntax. In PowerShell — the Windows 11 default — it fails **silently**, so the correct image was built and then never used. Fixed: docs give `$env:FENESTRA_IMAGE = "..."` first. |

**Consequence for your data.** None. This cannot produce a wrong number — the run stops. The cost
is time, not correctness.

**Which cards are affected.** Ask yours directly:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

| `compute_cap` | Architecture | Native install (torch 2.4.0+cu124) | Reference DL container (torch 1.14) | Standard all-in-one image | `cu128` all-in-one image |
|---|---|---|---|---|---|
| 12.0 | Blackwell (RTX 5060–5090) | ❌ no kernels | ❌ no kernels | ❌ no kernels | ✅ |
| 10.0 | Blackwell datacenter (B100, B200) | ❌ no kernels | ❌ no kernels | ❌ no kernels | ✅ |
| 7.0 – 9.0 | Volta → Hopper (RTX 20/30/40, A100, H100) | ✅ | ✅ up to 8.6 only | ✅ | ✅ |
| 5.0 – 6.x | Maxwell, Pascal (GTX 900, GTX 10-series) | ✅ | ✅ | ✅ | ❌ dropped |

The native install column is the host environment pinned in
[Host environment](../install/host-environment.md): torch 2.4.0+cu124, whose arch list is the same
`sm_50`–`sm_90` as the standard image. The reference DL container (`containers/dl_upsampling.def`,
`containers/Dockerfile`) is torch 1.14, which carries no PTX above `sm_86` — see
[issue 12](#12-the-all-in-one-image-is-not-the-reference-dl-stack). So on a Blackwell card the
`cu128` all-in-one image is the only route that works at all: the native conda + pip install and
both reference container recipes are excluded.

Both arch lists were measured, not assumed, with `torch._C._cuda_getArchFlags()` — which, unlike
`torch.cuda.get_arch_list()`, reports the compiled list even with no GPU attached:

```text
cu124 → sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90
cu128 → sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120
```

**Fix.** Build the Blackwell variant and select it. On Windows, in **one** PowerShell window:

```powershell
cd C:\FenestRA
docker build -t livrvub/fenestra:cu128 -f containers\Dockerfile.allinone.cu128 .
$env:FENESTRA_IMAGE = "livrvub/fenestra:cu128"
.\containers\run_fenestra.bat D:\path\to\your\scans
```

Confirmed working on Windows 11 with an RTX 5070 on 18 September 2026. The pre-flight arrives in
two stages. The launcher prints, in this order:

```text
Checking GPU access...
Found 3 scan(s) in D:\path\to\your\scans
Found 1 checkpoint(s) in D:\path\to\your\models
Image:            livrvub/fenestra:cu128
```

Then the container starts and prints its own banner — this takes a few seconds, and longer on the
first run:

```text
GPU: NVIDIA GeForce RTX 5070 (sm_120, CUDA 12.8)
```

`sm_120` there is the confirmation that the `cu128` image is doing its job. If the wrong image is
running you get the boxed `WARNING: ... reports sm_120, which this PyTorch build cannot run` block
instead. If any of these is not what you expect, stop and fix it before loading a scan.

**How to check what you are actually running**, since the tag alone is only a label:

```bash
docker exec fenestra /opt/venv-dl/bin/python -c "import torch; print(torch.__version__, torch._C._cuda_getArchFlags())"
```

`sm_120` in that output means your card is supported. No `sm_120` means it is not, whatever the
tag says.

!!! warning "The cu128 image is further from the validated stack"

    The reference backend is torch 1.14; the standard image is 2.1.2; this one is 2.8.0. Same
    architectures, same weights, different kernels. A Blackwell card **cannot** run the reference
    container at all, so "reproduce on the validated stack" and "use this GPU" are mutually
    exclusive. For numbers going into a manuscript, use `containers/dl_upsampling.def` on a card it
    supports. See [issue 12](#12-the-all-in-one-image-is-not-the-reference-dl-stack).

## Hardcoded settings that move your numbers

These are not defects, but they are fixed in the source, not exposed in the interface, and they change what you measure.

- **`min_size=15` pixels** in both Cellpose calls (`pipeline.py:253`, `:378`). Objects below 15 pixels in area are discarded. That is an equivalent diameter of about 4.4 pixels, which is roughly 27 nm on a 6.25 nm/px output grid and roughly 110 nm on a 25 nm/px grid. The cutoff in nanometers moves with your acquisition scale, so the smallest detectable pore is not constant across a study that mixes scan settings.
- **Per-image contrast normalization.** `normalize={"normalize": True, "percentile": (1.0, 99.0)}` is recomputed for every image (`pipeline.py:249`, `:374`). In a batch run each image gets its own stretch before segmentation, so rows in one `batch_results.xlsx` are not strictly comparable to each other.
- **`<stem>_upsampled.tif` has two possible meanings.** Straight from the backend it is float32 in physical height units (`inference.py:143`). With **Apply Post-DL Sharpening** ticked it is uint16 rescaled to the full 16-bit range (`pipeline.py:78`). Nothing in the filename or the workbook records which, so heights measured off those files depend on a checkbox. See [Outputs](../reference/outputs.md).
- **Tile blending is a boxcar average** over the 32-pixel overlap (`inference.py:181-184`), and HAT uses a 16-pixel window partition. Both can leave periodic structure in the output. See [The scale-domain question](scale-domain.md#tiling-and-window-artifacts).

## Almost no automated tests

Since 0.3.0 the repository contains two test files, both plain asserts: `tests/test_dl_cmd.py`, six checks that the three engines agree on the deep-learning command that `_build_dl_cmd` produces, and `tests/test_worker_errors.py`, four checks that `FenestraError` is not a `RuntimeError` subclass and that the `_reporting` decorator converts one — superqt swallows a `RuntimeError` raised inside a `@thread_worker` generator, producing no dialog at all. There is no test framework and no continuous integration, and neither file runs automatically. Run both by hand in the host environment:

```bash
python tests/test_dl_cmd.py
python tests/test_worker_errors.py
```

Nothing covers loading, upsampling, segmentation, quantification or the interface. A behavioral change in any of those is not caught by anything other than a person looking at the output.

If you upgrade FenestRA, re-run one scan you have already measured and confirm the diameters and porosity match your previous result before processing new data.

## Related pages

- [The scale-domain question](scale-domain.md) for the acquisition condition behind issue 5.
- [The all-in-one container](../install/all-in-one.md) for the image behind issue 12.
- [Troubleshooting](troubleshooting.md) if you are looking up an error string.
- [Parameters](../reference/parameters.md) for what every control does.
