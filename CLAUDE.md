# CLAUDE.md — FenestRA (napari_plugin)

Guidance for Claude Code when working in `napari_plugin/`. Part of the larger `AFM-Project/` repo.
Sibling project: `../DL_Upsampling/` — read its `CLAUDE.md` too; this plugin **ships its models
and inherits its failure modes**.

## Purpose

**FenestRA** — *Fenestration Resolution & Analysis* — is a napari plugin that turns the
`DL_Upsampling` research pipeline into five buttons for a biologist. It ingests a raw
`.jpk-qi-image` AFM scan of a Liver Sinusoidal Endothelial Cell, super-resolves it ×4 with the
project's HAT/SwinIR checkpoints, segments fenestrations with Cellpose, and exports physical
morphology metrics (area, perimeter, equivalent diameter, eccentricity, porosity) to CSV or a
consolidated XLSX.

Published on PyPI as **`napari-fenestra`** (current 0.2.11), BSD-3, Zenodo DOI
[10.5281/zenodo.19700659](https://doi.org/10.5281/zenodo.19700659), funded by MSCA
ImAge-d (grant 101119613). Remote: `https://github.com/LIVR-VUB/FenestRA.git`.

⚠️ **The model weights are deliberately NOT public** — the README's "Pre-Publication Notice"
says they ship with the manuscript. The public repo is scaffolding + container recipes. Any
change that assumes weights are downloadable is wrong.

## Repo Layout

```
napari_plugin/
├── src/fenestra/
│   ├── _widget.py          628 ln — Qt UI. Five numbered QGroupBoxes, ALL state on the widget
│   ├── pipeline.py         463 ln — orchestration: jpk load, CLAHE, DL subprocess (one argv
│   │                                builder, three engines), cellpose, regionprops, batch loop
│   ├── backend/inference.py 228 ln — the SR script. Runs in the container, or in /opt/venv-dl
│   ├── backend/__init__.py    0 ln — empty, but REQUIRED (makes backend a package in the wheel)
│   ├── napari.yaml               — npe2 manifest, one widget contribution
│   └── __init__.py          17 ln — exports FenestraWidget; __version__ via importlib.metadata
├── containers/
│   ├── dl_upsampling.def         — Apptainer/Singularity recipe (Linux / HPC). REFERENCE STACK
│   ├── Dockerfile                — same recipe, Docker port (Windows / macOS). REFERENCE STACK
│   ├── Dockerfile.allinone       — ⭐ v0.3.0. ONE image: napari + plugin + backend + noVNC
│   ├── entrypoint.sh             — Xvfb → openbox → x11vnc → websockify → napari
│   ├── fenestra-app.py           — opens napari with the FenestRA dock already docked
│   └── run_fenestra.{bat,sh}     — one-click launchers, loopback-only port publish
├── tests/test_dl_cmd.py          — 6 plain asserts over _build_dl_cmd. No framework. Run:
│                                   python tests/test_dl_cmd.py
├── misc/                         — FenestRA.jpg logo, eu_funded.jpg
├── setup.cfg                     — ALL packaging metadata lives here (setup.py is a 3-line stub)
├── pyproject.toml                — build-system only
└── README.md                     — install + usage + changelog + citations
```

**No `.github/`. No git tags.** `tests/test_dl_cmd.py` exists as of 0.3.0, but it is one file
covering one function and **nothing runs it automatically**. If you add anything behavioural,
there is still no safety net — say so rather than assuming CI catches it.

`dist/` and `src/napari_fenestra.egg-info/` exist on disk but are **gitignored and untracked**
(`git check-ignore -v` confirms `.gitignore:7:dist/`). They are local build residue. Do not
"restore" them to git.

## Branches

| Branch | Tip | Meaning |
|---|---|---|
| `main` | `edd45c0` 2026-04-23 | current, **in sync with `origin/main`** |
| `v0.2` | `78e4ea4` 2026-04-22 | frozen release snapshot |
| `v0.1` | `e4ecfed` 2026-04-18 | frozen release snapshot |

`v0.1 → v0.2` added the batch module + post-DL sharpening (3 files, +523 lines).
`v0.2 → main` added the LICENSE, the embedded `backend/inference.py`, and the npe2 name fixes.
39 commits total, all 2026-04-17 → 2026-04-23. **Do not commit to `v0.1`/`v0.2`** — they are
release records.

## Architecture — hub and spoke

The one genuinely load-bearing design decision. Two Python environments that never share a
process:

| | Host / GUI | Backend |
|---|---|---|
| Base | conda `fenestra-env`, py3.10 | `nvcr.io/nvidia/pytorch:23.01-py3` = **py3.8 / torch 1.14** |
| Holds | napari, Qt, **Cellpose 4**, AFMReader, torch 2.4/cu124 | basicsr, HAT, SwinIR (git-cloned to `/opt`), opencv-headless 4.8.0.74, numpy<1.24 |
| Why | modern GUI stack | basicsr will not coexist with modern numpy/torch |

The **entire** contact surface is one `subprocess.run` + a temp `.tif` on disk. As of 0.3.0 it is
built in exactly one place, `_build_dl_cmd()` at `pipeline.py:109`, and run in one place,
`_run_dl_inference()` at `:168`.

### The separation is a *process* boundary, not a container boundary

That distinction became load-bearing in 0.3.0. The spoke can now be reached three ways:

| Engine | Spoke is | Contact surface |
|---|---|---|
| `singularity` | the `.sif` | `singularity exec --nv` + 4 `--bind` |
| `docker` | the tagged image | `docker run --gpus all` + 4 `-v` |
| `local` | **`/opt/venv-dl` on the same filesystem** | a plain subprocess, real paths, no mounts |

`local` exists because **a container cannot launch a container**. Once the plugin itself is
containerised — `containers/Dockerfile.allinone` — the `docker run` spoke has nothing to run on.
So the second environment moves inside the image as a second venv, and the isolation is carried
by the venv rather than by the container. Same guarantee, one less layer.

Because both venvs live in one filesystem, the thing that would break the isolation is an
inherited interpreter variable. `_build_dl_cmd()` therefore strips `PYTHONPATH` and `PYTHONHOME`
from the `local` subprocess environment. HAT and SwinIR reach the DL interpreter through a `.pth`
file in its `site-packages` instead, precisely so no `PYTHONPATH` is needed.

⚠️ The all-in-one image's DL venv is **torch 2.1.2**, not the reference stack's 1.14 — torch 1.13
and 1.14 wheels carry no PTX and refuse to start above sm_86, which excludes every RTX 40-series
card and H100. The reference recipes are unchanged and remain the provenance for published
numbers. This is open item 13.

⚠️ The PyTorch **2.4** badge in the README describes the *host*. The container is **torch 1.14**.
Same fact as [[afm-unified-container]] in the sibling project.

Commit `4a28086` moved `inference.py` *into* the pip package. Before that the plugin needed the
`DL_Upsampling` repo checked out locally. `backend_dir` is resolved as
`os.path.dirname(__file__)/backend` — i.e. **from inside site-packages** — and bind-mounted to
`/opt/dl_project/scripts`. Verified present in the built wheel:
`fenestra/backend/inference.py`, `fenestra/backend/__init__.py`, `fenestra/napari.yaml` all
ship. Do not "clean up" `backend/__init__.py` — deleting it drops the directory from the wheel.

### The all-in-one image's DL venv, measured 2026-09-18

Built from `containers/Dockerfile.allinone`. Do not re-derive these:

- `pip install basicsr==1.4.2` **upgrades numpy to 2.2.6** on the way in, because basicsr's
  requirements say only `numpy>=1.17`. The following `--force-reinstall --no-deps "numpy<1.24.0"`
  puts it back to **1.23.5**, which is what the image ships. Reorder those two lines and the venv
  silently ends up on numpy 2.
- That leaves `scikit-image 0.25.2` installed while declaring `numpy>=1.24`. **It is inert.**
  basicsr never imports skimage — the only occurrence of the string in the entire package is a URL
  in a comment at `basicsr/data/degradations.py:563`. `pip check` would complain; nothing breaks.
- `scipy` **is** imported (`basicsr/data/degradations.py` → `from scipy import special`, reached
  via `basicsr/__init__.py` → `from .data import *`). scipy 1.15.3 requires `numpy>=1.23.5` and
  gets exactly 1.23.5. Exactly satisfied, with no margin — a scipy that raises its floor breaks
  the build.
- The build-time smoke test is what catches all of this. It imports numpy, torch, tifffile, tqdm,
  einops, timm, `torchvision.transforms.functional_tensor.rgb_to_grayscale`, `basicsr`, the HAT
  class loaded by absolute path, and both SwinIR sources. Measured result:
  `DL venv OK - torch 2.1.2+cu121, numpy 1.23.5`. Keep that test; it is the only guard.

## Dataflow

```
.jpk-qi-image
  └─ AFMReader.load_jpk(channel="height_trace", flip_image=True) → (array, nm/px)
      ├─ CLAHE path (CPU): scipy.ndimage.zoom(order=3, ×factor)
      │                    → equalize_adapthist → unsharp_mask → uint16
      └─ DL path: write temp.tif → singularity exec --nv | docker run --gpus all
                    inference.py: per-image MIN-MAX → [0,1]
                                  → reflect-pad to multiple of window_size
                                  → HAT/SwinIR ×4, tiled at 256, overlap 32
                                  → unpad → np.clip(0,1)
                                  → ×(vmax−vmin)+vmin → float32 .tif
                  → optional CLAHE + unsharp  (⚠️ changes dtype AND units, see below)
  → Cellpose on the HOST GPU (custom .pth, else cpsam) → instance labels
  → regionprops → Area_nm² / Perimeter_nm / Equivalent_Diameter_nm / Eccentricity
  → porosity = Σ mask area / image area
  → CSV (single image) | batch_results.xlsx + per-image _upsampled.tif and _mask.tif (batch)
```

Two execution modes share the physics. `run_dl_upsampling` / `run_cellpose` /
`run_batch_pipeline` are `@thread_worker` generators for the GUI; `run_dl_upsampling_sync` /
`run_cellpose_sync` are **verbatim copies** used by the batch loop.

~~⚠️ The container argv block is duplicated byte-for-byte at `pipeline.py:68-99` and `:226-257`.~~
**Fixed in 0.3.0.** There is now one `_build_dl_cmd()` returning `(argv, env)` and one
`_run_dl_inference()` that runs it and returns the output TIFF path. Both the `@thread_worker`
path and the batch path call it, so the two cannot diverge. `tests/test_dl_cmd.py` asserts that
the Docker and Singularity forms still pass an identical inner command.

## Where this sits relative to DL_Upsampling

FenestRA is the **deployment** of the v6 models. It inherited the science and almost none of the
guardrails.

| `DL_Upsampling` has | FenestRA has |
|---|---|
| `src/methods/checkpoint_io.py` (20 KB) — refuses wrong arch, wrong scale, truncated file, 3-channel ckpt | `load_state_dict(strict=True)` and nothing else |
| architecture derived from the checkpoint's own tensor shapes | architecture **hardcoded** in `build_model()` |
| per-scan nm/px carried through the whole pipeline | reads nm/px, then **never checks it** |
| store-time percentile normalisation, unclipped | per-image min-max, clipped to [0,1] |
| documents that 7/18 checkpoints on disk are sign-inverted | no detection |
| `scripts/infer/inference.py` — **564 lines**, config-driven, JPK input, classical baselines | a **228-line fork**, frozen 2026-04-23 |

`backend/inference.py` is a **stale simplified fork** of `DL_Upsampling/scripts/infer/inference.py`.
The upstream has since grown config support, direct `.jpk` ingestion, bicubic/lanczos arms and
the `checkpoint_io` guards (upstream mtime 2026-08-06 vs the fork's 2026-04-23). **They are not
kept in sync and there is no mechanism that would notice.** Before touching the fork, diff it:
`diff -u ../DL_Upsampling/scripts/infer/inference.py src/fenestra/backend/inference.py`.

---

# Measured facts — do not re-derive

Everything in this section was measured on 2026-08-11 against the real files and the real
installed environment. Cite these rather than re-running.

## 1. The plugin can only load HALF the shipped v6 arms

`backend/inference.py:47-71` hardcodes `embed_dim=180, depths=[6]*6, num_heads=[6]*6` for both
architectures (`window_size` 16 for HAT, 8 for SwinIR), then loads with `strict=True`.

From `../DL_Upsampling/configs/train/`:

| Arm | embed_dim | depths | loads in FenestRA? |
|---|---|---|---|
| `hat_v6` (A), `hat_v6b` | 180 | [6]×6 | ✅ |
| `swinir_v6` (A), `swinir_v6b` | 180 | [6]×6 | ✅ |
| **`hat_v6c_small`** | **96** | **[6]×4** | ❌ **raises** |
| **`swinir_v6c_small`** | **96** | **[6]×4** | ❌ **raises** |

`swinir_v6c` is a **live, healthy arm** (37.51 dB val, best attribution concentration in the xAI
suite). A user selecting it gets a `RuntimeError` from `load_state_dict`, surfaced as a
`QMessageBox` reading `Container DL Inference failed: <stderr>`. This is the *good* failure mode —
loud — but the plugin cannot use the small models at all.

`strict=True` is the single most valuable line in this file. **Never relax it to
`strict=False`.** The sibling project's characteristic bug is *plausible output from the wrong
network*, and `strict=False` is exactly how that happens.

## 2. The input transform does not match training, and the output is clipped

Measured from `Dataset-3/store_v6/manifest.json` (130 scans):

```
normalisation: { method: percentile, lo_pct: 0.1, hi_pct: 99.9, clipped: false }
hr_max_norm:  min 1.006  median 1.040  max 1.890   →  130/130 scans exceed 1.0
hr_min_norm:  min −0.198 median −0.032 max 0.000   →  124/130 scans go below 0.0
```

Training deliberately does **not** clip. The plugin does two different things:

- `inference.py:98-106` — per-image **min-max** to exactly [0,1]. Not the percentile transform
  the model was trained under, and one bright speck rescales the whole scan.
- `inference.py:138` — `np.clip(out_npy, 0, 1)` **before** restoring physical height.

The clip truncates precisely the tails the network was trained to produce. Same defect the
sibling project already recorded for `results/inference_v6/` ("clipped to [0,1] while HR runs to
1.139, biasing any per-pore metric exactly at the rim"). Here it matters more, because the
clipped array is what Cellpose segments and what `regionprops` measures — so the bias lands
directly in the published diameters and porosity.

## 3. The scale-domain question — the most important scientific caveat

Training HR is **median 25.00 nm/px** (range 15.62–42.86, n=130). The model was trained to invert
a ×4 synthetic degradation, so the LR it learned on is **median 100 nm/px**.

The plugin feeds the raw loaded `.jpk` array straight into the ×4 network. **There is no check on
`pixel_to_nm` anywhere before quantification** — grep `_widget.py` and `pipeline.py`: the value is
read at `pipeline.py:57-62`, stored, and next used only at `_widget.py:542` /
`pipeline.py:509` to convert pixels to nanometres.

So if a user loads a scan acquired at normal settings (~25 nm/px), the network is being asked to
upsample something **4× finer than anything it saw in training**, and will report diameters on a
6.25 nm/px grid. Nothing warns. Nothing fails. The output looks like a plausible AFM image.

⚠️ **This is the plugin-side face of the synthetic-degradation circularity** that gates
publication for the whole project ([[publication-strategy]]). Do not paper over it with a
warning dialog and call it solved — the honest fix is a documented acquisition protocol
(acquire at ~100 nm/px, upsample to ~25) plus a guard that refuses or warns outside that band.

The arithmetic itself is correct: `upsampled_scale = pixel_to_nm / factor`, then
`diameter_nm = equivalent_diameter_px × upsampled_scale`. Worked example: 100 nm/px input,
factor 4 → 25 nm/px output; a pore 8 output-px across → 200 nm. Right direction.

## 4. Cellpose 4 changes the meaning of two parameters, silently

`fenestra-env` has **cellpose 4.1.1** (also on this machine: 4.0.4, 3.1.1.2, 2.3.2 in other
envs). Read from the installed `cellpose/models.py`:

- **`model_type` is accepted and IGNORED.** `models.py:108-110`:
  ```
  if model_type is not None:
      ... "model_type argument is not used in v4.0.1+. Ignoring this argument..."
  ```
  **Fixed in 0.3.0 — as a labelling fix, not a behaviour change.** The ignored
  `model_type="cyto2"` argument was deleted from both cellpose calls, and the UI placeholder now
  reads *"Leave empty for the Cellpose 4 default (cpsam)"*. The user still gets `cpsam`; the
  difference is that the code and the UI now say so. `setup.cfg` pins `cellpose>=4.0.1` because
  of it — on cellpose 2 or 3, `CellposeModel(gpu=...)` with no `model_type` has no model to
  load.

- **`diameter` no longer means what the label says.** `models.py:272-273`:
  ```
  if diameter is not None and diameter > 0:
      image_scaling = 30. / diameter
  ```
  The UI default is `30.0` → scaling `1.0` → **no-op**. The label read *"Diameter (0=auto)"*
  until 0.3.0 and now reads *"Diameter (30 = no rescale)"*; `0` fails the `> 0` test and produces no
  rescale. **0 and 30 are the same setting.** There is no "auto" in cellpose 4.

- **`tile=True` does not exist in cellpose 4.1.1's `eval()`.** Confirmed against the real
  signature (which has `tile_overlap` and `bsize`, no `tile`). The plugin's retry loop
  (`pipeline.py:265-282`, `:389-406`) catches the `TypeError`, regex-extracts the bad kwarg,
  drops it and retries. It works — but by accident, and it means every run does a wasted first
  `eval()`. ⚠️ **The loop only wraps `model.eval`, never the constructor**, so a constructor-level
  API break is unguarded.

- **`use_bfloat16=True` is the default** (`models.py:92`, dtype set at `:143`). The plugin never
  passes it. This is the README's Maxwell/Pascal `CUBLAS_STATUS_NOT_SUPPORTED` warning.
  Commit `f1d277a` set it `False`; commit `edd45c0` **reverted that** to keep bf16 acceleration
  on modern GPUs. That revert is deliberate — the README documents the limitation instead.
  Do not "fix" it back without asking.

- **`normalize={"normalize": True, "percentile": (1.0, 99.0)}` is PER-IMAGE.** In a batch run,
  every image gets its own contrast stretch, so segmentations are **not comparable across images
  in the same `batch_results.xlsx`**. The sibling project hit this and settled the opposite way —
  `phantom_probe.yaml` pins `fixed_normalisation: true` precisely so the readout does not move
  per frame ([[phantom-sweep-design]]). The batch module has the bug that experiment was
  designed to avoid.

- `min_size=15` (px) is hardcoded in both cellpose calls. At a 6.25 nm/px output grid that
  excludes anything under ~27 nm equivalent diameter; at 25 nm/px, ~110 nm — i.e. **it silently
  eats the bottom of the biological range** depending on input scale. The sibling project lowered
  its equivalent to 4 and recorded it as a protocol deviation.

## 5. The Docker engine path — fixed 2026-09-18, was broken

`containers/Dockerfile` used to end `ENTRYPOINT ["python"]` while `pipeline.py:145-161`
builds:
```
docker run --rm --gpus all -v ... <image> python /opt/dl_project/scripts/inference.py --input ...
```
Docker concatenates ENTRYPOINT + CMD, so the argv inside the container was
**`python python /opt/dl_project/scripts/inference.py ...`** — Python treats the literal string
`python` as the script path and dies with
`can't open file '/opt/python': [Errno 2] No such file or directory`.

**Resolved by deleting the `ENTRYPOINT` line**, not by setting `ENTRYPOINT []`. Deleting inherits
the NGC base image's own entrypoint, which execs the given command; `ENTRYPOINT []` would clear it.
`pipeline.py` was deliberately left alone — fixing the argv there would have meant editing two
duplicated blocks, and the recipe is the single place the asymmetry belongs.

*Confidence: reasoned from documented ENTRYPOINT/CMD semantics and the two source files.
Neither the failure nor the fix was executed — the image is not built on this machine.*
⚠️ **Images built before this change still carry the bad ENTRYPOINT and still fail.** A rebuild is
required, which is why the docs now frame the error as a stale-image symptom rather than a defect.

The Singularity path was always **correct**: `singularity exec` bypasses `%runscript`, so the
explicit `python` is required there. The two engines genuinely need different argv, which is why
this slipped through and why only one recipe had to change. Do not "unify" them.

Related, unverified but load-bearing for the README's cross-platform claim:
- `--gpus all` has no meaning on Docker Desktop for macOS (no NVIDIA passthrough). The README
  advertises macOS.
- The `-v` mounts pass host paths straight through. On Windows those are `C:\Users\...`,
  including `tempfile.gettempdir()` and the site-packages `backend_dir`.
- `livrvub/dl-upsampling:latest` is a **locally built tag**, not a Docker Hub image — the README
  tells the user to build it. If it is ever pushed, the README build step becomes optional.

## 6. Packaging

Correct: the wheel ships `fenestra/napari.yaml`, `fenestra/backend/inference.py` and
`fenestra/backend/__init__.py`. `[options.package_data] * = *.yaml` + `include_package_data` is
enough because the backend is `.py` inside the package. METADATA version 0.2.11 matches
setup.cfg.

Both former defects are fixed in 0.3.0:

- ~~`src/fenestra/__init__.py:1` says `__version__ = "0.0.1"`.~~ It now reads the installed
  version with `importlib.metadata.version("napari-fenestra")`, falling back to
  `"0.0.0+unknown"` only in an uninstalled source tree. `setup.cfg` is still the single source of
  truth; there is no longer a second copy to drift.
- ~~`install_requires` omits three runtime imports.~~ **`AFMReader` turned out to be on PyPI**
  (`AFMReader==0.0.7`, identical `load_jpk` signature to the git build), so it is now declared,
  along with the `pySPM<0.6.3` pin it needs — 0.6.3 requires numpy≥2. That also removes Git as a
  Windows prerequisite entirely. `cellpose` is now pinned `>=4.0.1`.

  Two omissions REMAIN, deliberately, and are documented in the README rather than fixed:
  **`napari`** (a napari plugin conventionally does not depend on napari; the viewer and its Qt
  backend are the user's choice) and **`torch`** (it arrives via `cellpose`, and declaring it
  would not change which build pip fetches — only the ordered install in README step 2 secures
  the CUDA wheel).
- `pyproject.toml` requires `setuptools_scm[toml]>=3.4` but `setup.cfg` pins a static version and
  **there are no git tags** (`git tag -l` → empty). It does nothing. Harmless leftover.

---

# Silent-failure mechanisms to remember

Same house rule as the sibling project: **this codebase's characteristic bug is plausible output,
never an exception.** Ranked by how likely a biologist is to publish the wrong number.

1. **Cellpose `model_type="cyto2"` is ignored → cpsam runs instead** (§4). No error, plausible
   masks, and the UI actively tells the user the opposite.
2. **No scale check before inference** (§3). A scan at the wrong nm/px produces a beautiful,
   entirely out-of-domain super-resolution.
3. **Sign-inverted checkpoints load cleanly.** The sibling project documents **7 of 18
   checkpoints on disk as sign-inverted** ([[ffl-phase-blind-bug]]) — the network outputs the
   photographic negative. `inference.py:141` rescales to `[vmin, vmax]`, so an inverted output
   is still a plausible-looking AFM height map. The plugin has **no detection**. Cheapest guard:
   `pearson(bicubic_upsample(input), output)` and refuse below 0. One line, catches every one of
   the seven.
4. **`finalize_upsampling()` hardcodes `scale=(0.25, 0.25)`** (`_widget.py:455`, and again for
   the mask at `:487` and the overlay at `:522`) with the comment `# assumed 4x upsampling`.
   Correct for DL. **Wrong for CLAHE at any factor except 4** — the Factor spinbox ranges 1–10.
   The viewer then shows Raw and Upsampled at mismatched physical scales in the 4-pane grid, and
   they look aligned.
5. **`on_quantify()` reads the LIVE dropdown, not what was run** (`_widget.py:539-540`):
   ```
   method = self.combo_method.currentText()
   factor = self.spin_up_factor.value() if "CLAHE" in method else 4.0
   ```
   Run CLAHE at factor 2 → switch the Method dropdown to HAT → click Quantify → every diameter
   is reported **2× too small**, with no warning. The widget never records which method actually
   produced `self.upsampled_image`. Fix: store the factor on the widget at
   `finalize_upsampling()` time.
6. **`apply_post_processing()` changes dtype AND units.** It returns **uint16 normalised**
   (`pipeline.py:78`), while `inference.py` returns **float32 in physical height units**. So
   `<name>_upsampled.tif` from a batch run means two different things depending on whether the
   "Apply Post-DL Sharpening" checkbox was ticked, with nothing in the filename or the XLSX
   recording which. Anyone measuring heights off those TIFFs is measuring a checkbox.
7. **The tile blend is a boxcar, not feathered.** `inference.py:181-184` accumulates whole tiles
   and divides by a coverage count — a hard rectangular average over the 32-px overlap. That
   leaves seams. Note the sibling project already measured that **HAT's 16-px window partition
   leaves a periodic fingerprint** (modulation 1.9× its coprime control) — so periodic texture in
   a FenestRA output has *two* possible non-biological sources. Do not read fine periodic
   structure in a FenestRA image as biology.
8. **`--tile_size 256` is hardcoded** in both argv blocks, so tiling runs even on small images
   that would fit whole. The arithmetic is currently safe — `process_image()` pads the full
   image to a multiple of `window_size` first, and `stride = 256 − 32 = 224 = 16 × 14`, so every
   tile is a multiple of 16 and `pad_h`/`pad_w` are always 0. ⚠️ **That safety is accidental.**
   Change `tile_size` or `overlap` to anything that breaks `stride % 16 == 0` and the per-tile
   reflect-pad at `inference.py:170` can be asked to pad more than the tile dimension, which
   torch forbids and which raises.
9. ~~**Hardcoded developer paths survive in the shipped UI.**~~ **Fixed in 0.3.0.** Commit
   `f724562` had fixed `pipeline.py` and missed `_widget.py`, leaving
   `/home/arka/Desktop/AFM-Project/DL_Upsampling/...` pre-filled in two text boxes of a published
   package. The defaults now come from `FENESTRA_DL_MODEL`, `FENESTRA_SIF`,
   `FENESTRA_DOCKER_IMAGE`, `FENESTRA_ENGINE` and `FENESTRA_CP_MODEL`, defaulting to empty, and
   `on_engine_changed` keeps a per-engine value instead of overwriting what the user typed.
   `grep -c /home/arka src/fenestra/_widget.py` → 0.

---

# Conventions when editing here

- **One container argv builder — keep it that way.** `_build_dl_cmd()` is the only place that
  writes the DL command line, and `_run_dl_inference()` the only place that runs it. Both the
  interactive worker and the batch loop go through them. Adding a fourth engine, or changing
  mounts, flags or `--tile_size`, means editing exactly one function. Do not re-introduce a
  second copy, and keep `tests/test_dl_cmd.py` passing — it asserts the Docker and Singularity
  forms still produce an identical inner command.
- **Three engines, not two.** `singularity`, `docker` and `local`. `_engine_key()` normalises the
  UI label (`"Local (bundled)"` → `"local"`), so a new engine needs a keyword that survives
  `.strip().lower().split()[0]`. The `local` engine exists because a container cannot launch a
  container — it is what makes `Dockerfile.allinone` possible — and it scrubs `PYTHONPATH` and
  `PYTHONHOME` from the subprocess so the GUI venv cannot leak into the DL venv.
- **Never raise `RuntimeError` inside a `@thread_worker`.** Raise `FenestraError` (or let
  `@_reporting` convert it). Measured against the installed superqt: `GeneratorWorker.work()`
  catches `RuntimeError` and *returns* it, then `WorkerBase.run()` warns and returns **before**
  `errored` and `finished` fire. The result is this project's worst failure shape — no dialog, the
  button frozen at "Upsampling in progress...", `is_running` stuck `True`, and the user has to
  restart napari to learn anything went wrong. Reproduced directly: a worker raising
  `RuntimeError` emits *no signals at all*, while the same worker raising `ValueError` emits
  `errored` normally. `tests/test_worker_errors.py` guards the one invariant that matters,
  `FenestraError` not being a `RuntimeError` subclass. This also covers library `RuntimeError`s
  raised inside a worker, notably CUDA out-of-memory from torch or cellpose.
- **`strict=True` in `build_model()` stays.** It is the only thing standing between a user and a
  confidently-wrong network.
- **Do not widen `except`.** `pipeline.py`'s cellpose retry loop already re-raises any `TypeError`
  it cannot parse, and the HAT import in `inference.py:33-45` falls back on a broad `except
  Exception`. The sibling project's `build_model()` bug — a broad except swallowing a registry
  error and returning a **SwinIR labelled as HAT** — is the same shape. Narrow it, don't widen it.
- **`.tif` via `tifffile`, never PIL/cv2** — 16-bit and float32 AFM height data must survive.
- **Anything user-facing that names a Cellpose model or a diameter is a claim about cellpose's
  API**, and that API changed under this code (§4). Check the installed version before trusting
  a label.
- **Version bumps touch `setup.cfg` only.** `src/fenestra/__init__.py` reads it back through
  `importlib.metadata`; do not add a second literal. The README changelog is hand-maintained.
- **Adding a runtime import means updating `install_requires`** in `setup.cfg`, or documenting
  in the README why it cannot be declared (as with `napari` and `torch`).
- **Three container recipes now, and two of them are near-duplicates.**
  `containers/Dockerfile.allinone.cu128` is `Dockerfile.allinone` with four lines changed: both
  venvs' torch/torchvision (→ 2.8.0/0.23.0 cu128), the DL venv's numpy ceiling (<1.24 → <2), and a
  one-line `sed` patching `basicsr/data/degradations.py` — which the newer torchvision makes
  necessary and which BasicSR itself already fixed upstream. It exists because RTX 50-series cards
  are **sm_120** and the standard image's torch carries no kernels for them:
  `torch.cuda.is_available()` is `True`, then the first launch raises
  `CUDA error: no kernel image is available for execution on the device`.
  ⚠️ **The two files will drift.** Before editing either, `diff -u containers/Dockerfile.allinone
  containers/Dockerfile.allinone.cu128` — the diff should stay at those four changes plus the
  cu128 file's extra build-time assertions.
  Measured arch flags (`torch._C._cuda_getArchFlags()`, which works with no GPU attached, unlike
  `torch.cuda.get_arch_list()`):
  `cu124 → sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90`;
  `cu128 → sm_70 sm_75 sm_80 sm_86 sm_90 sm_100 sm_120`. So cu128 gains Blackwell and **loses
  Maxwell and Pascal**.
- **Two container recipes with different jobs.** `dl_upsampling.def` / `Dockerfile` build the
  **reference stack** (torch 1.14, nvcr 23.01) and are what publication numbers should come from.
  `Dockerfile.allinone` is the **deployment** image (torch 2.1.2, because 1.14 wheels carry no
  PTX and will not start above sm_86). A change to the science belongs in the reference recipes
  first.
- When a fix depends on something in `../DL_Upsampling/`, **read that repo's `CLAUDE.md` first** —
  most of the traps here are already characterised there under different filenames.

## Open items (as of 2026-09-18)

Ordered by consequence for a published number, not by effort.

1. **Sign-inversion guard** — §silent-failure 3. One correlation check. Cheapest defence against
   the sibling project's most-repeated bug.
2. **Scale-domain guard + documented acquisition protocol** — §3. The gating scientific issue.
3. ~~**Fix the Docker argv** — §5.~~ Done 2026-09-18: `ENTRYPOINT` deleted from the Dockerfile.
   Not executed end-to-end; the Windows/macOS path is unverified, not proven working.
4. **Fix `on_quantify()` factor desync and the hardcoded `scale=(0.25,0.25)`** — §silent-failure
   4 and 5. Both produce wrong physical sizes from a correct image.

   ⚠️ **Re-confirmed by adversarial review on 2026-09-18 and deliberately NOT applied**, at the
   maintainer's instruction, to keep 0.3.0 scoped to containerisation. Both faces share one root
   cause — *the widget never records the factor that actually ran* — so one attribute fixes both.
   The fix, ready to apply:

   - `__init__`, near `self.upsampled_image = None`: add `self.up_factor = 4.0`.
   - `on_run_upsampling`, CLAHE branch: `self.up_factor = float(factor)`.
   - `on_run_upsampling`, DL branch, before the worker starts: `self.up_factor = 4.0`.
   - Replace all three `scale=(0.25, 0.25)` literals with `scale=(1.0 / self.up_factor,) * 2`.
   - Replace the live-dropdown read in `on_quantify` with `factor = self.up_factor`.

   Reproduction, in case anyone doubts it is real: load a JPK, Method **CLAHE (CPU)**, Factor 2,
   Run Upsampling → Run Cellpose, then switch Method to **HAT** *without re-running* and click
   Quantify. Every diameter is reported 2× too small and every area 4× too small, silently. The
   factor is re-derived from whatever the dropdown reads at click time.

5. **Validate the Cellpose checkpoint path** — a non-empty path that does not exist falls back to
   `cpsam` and segments anyway. A trailing space copied from a file manager is enough, as is
   starting the all-in-one container without `-v <models>:/models`. Since 0.3.0 the fallback at
   least *prints* the path it failed to find, so it is visible in the terminal, but it does not
   stop. **Reviewed 2026-09-18, not applied** by instruction. The fix belongs in the shared
   helpers, not the caller, so the batch path is covered at the same time — in both `run_cellpose`
   and `run_cellpose_sync`, immediately before the `if os.path.exists(model_path)`:

   ```python
   model_path = (model_path or "").strip()
   if model_path and not os.path.exists(model_path):
       raise FileNotFoundError(f"Cellpose checkpoint not found: {model_path!r}")
   ```

   An empty field still means cpsam. `FileNotFoundError` is not a `RuntimeError`, so it reaches
   the dialog without needing `FenestraError`.
6. ~~**Relabel or re-implement the Cellpose model/diameter controls** — §4.~~ Relabelled in
   0.3.0, and `model_type="cyto2"` removed. The *behaviour* is unchanged and still surprising:
   an empty box gives cpsam, and 0 and 30 are the same diameter. Re-implementing remains open.
7. **Pin Cellpose normalisation for batch runs** — §4. Otherwise `batch_results.xlsx` rows are
   not comparable to each other.
8. **Support `embed_dim: 96` / `depths [6]×4`** so `swinir_v6c` and `hat_v6c` load — §1. Best
   done by deriving the architecture from the checkpoint the way
   `../DL_Upsampling/src/methods/checkpoint_io.py` does, rather than adding a second hardcoded
   branch.
9. ~~**Strip the `/home/arka/` defaults from `_widget.py`**~~ — done 0.3.0.
10. ~~**Reconcile `__version__`** and decide what `install_requires` should honestly claim~~ —
   done 0.3.0. See §6.
11. **De-duplicate the ~~container argv~~ and the sync/async cellpose twins.** The argv half is
    done (`_build_dl_cmd`). `run_cellpose` and `run_cellpose_sync` are **still verbatim twins**,
    including the `eval()` retry loop. Same fix, not yet applied.
12. **Real tests.** `tests/test_dl_cmd.py` and `tests/test_worker_errors.py` cover
    `_build_dl_cmd` and the worker-error contract only, and nothing runs either automatically.
    The round-trip test that would catch item 4 — synthetic image → CLAHE → cellpose stub →
    regionprops → known area — still does not exist.
13. ~~**Verify the all-in-one image end to end.**~~ **Largely done, 2026-09-18**, on Windows 11
    with an RTX 5070, using the `cu128` variant. Confirmed working by the user: `docker build`,
    the launcher, GPU passthrough through WSL 2, the browser desktop, loading a `.jpk-qi-image`,
    and **HAT upsampling running on the GPU**. The known-good invocation, now in the README:

    ```powershell
    cd C:\FenestRA
    $env:FENESTRA_IMAGE = "livrvub/fenestra:cu128"
    .\containers\run_fenestra.bat D:\path\to\your\scans
    ```

    ⚠️ Still unconfirmed: a full run through to **Quantify and a written CSV**, and the batch
    module. Do not record this as fully end-to-end verified until someone has the CSV.

    Three things that cost that user real time, all now fixed, all worth remembering:
    - **PowerShell is not Command Prompt.** `set FOO=bar` is silently a no-op in PowerShell, which
      is the Windows 11 default shell. Every Windows instruction must give `$env:FOO = "bar"` too.
    - **Two images look identical from the outside.** The launcher now prints its tag, because the
      only other way to tell them apart was spotting a `functional_tensor` path inside a 200-line
      traceback.
    - **An empty file dialog has three different causes.** Both launchers now count scans and
      checkpoints on the host, before starting, and name the exact folder.
14. **Decide whether the all-in-one's torch 2.1.2 backend is acceptable for publication**, or
    whether the reference stack needs a modern-GPU port of its own. Right now the honest answer
    is "use the reference container for numbers", which is a documentation fix, not a solution.

    ⚠️ **This got sharper on 2026-09-18.** A user's RTX 5070 cannot run the reference stack *or*
    the all-in-one image, so `Dockerfile.allinone.cu128` (torch 2.8.0) now exists. The gap between
    "the stack the method was validated on" and "the stack a current GPU can execute" is now
    1.14 → 2.8, and it widens with every GPU generation. Someone has to decide whether the
    reference stack gets re-validated on a modern torch, because telling users to produce
    publication numbers on hardware that is increasingly hard to buy is not a durable answer.

15. **`opencv-python-headless==4.8.0.74` is yanked on PyPI** ("deprecated, use 4.8.0.76"). Pinned
    exactly, so pip still installs it, in all three recipes. It works today and a yanked release
    can be removed at any time, which would break every build at once.
