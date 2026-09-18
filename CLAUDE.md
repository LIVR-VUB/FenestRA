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
│   ├── _widget.py          570 ln — Qt UI. Five numbered QGroupBoxes, ALL state on the widget
│   ├── pipeline.py         441 ln — orchestration: jpk load, CLAHE, container subprocess,
│   │                                cellpose, regionprops, batch loop
│   ├── backend/inference.py 228 ln — the SR script that runs INSIDE the container
│   ├── backend/__init__.py    0 ln — empty, but REQUIRED (makes backend a package in the wheel)
│   ├── napari.yaml               — npe2 manifest, one widget contribution
│   └── __init__.py           7 ln — exports FenestraWidget; ⚠️ __version__ is STALE (see below)
├── containers/
│   ├── dl_upsampling.def         — Apptainer/Singularity recipe (Linux / HPC)
│   └── Dockerfile                — same recipe, Docker port (Windows / macOS)
├── misc/                         — FenestRA.jpg logo, eu_funded.jpg
├── setup.cfg                     — ALL packaging metadata lives here (setup.py is a 3-line stub)
├── pyproject.toml                — build-system only
└── README.md                     — install + usage + changelog + citations
```

**No `tests/`. No `.github/`. No git tags.** Verified 2026-08-11. If you add anything
behavioural, there is no safety net — say so rather than assuming CI catches it.

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

| | Host | Container |
|---|---|---|
| Base | conda `fenestra-env`, py3.10 | `nvcr.io/nvidia/pytorch:23.01-py3` = **py3.8 / torch 1.14** |
| Holds | napari, Qt, **Cellpose 4**, AFMReader, torch 2.4/cu124 | basicsr, HAT, SwinIR (git-cloned to `/opt`), opencv-headless 4.8.0.74, numpy<1.24 |
| Why | modern GUI stack | basicsr will not coexist with modern numpy/torch |

The **entire** contact surface is one `subprocess.run` + four bind mounts + a temp `.tif` on
disk. That is the whole hub-and-spoke: `pipeline.py:66-99` (async) and `:224-257` (sync).

⚠️ The PyTorch **2.4** badge in the README describes the *host*. The container is **torch 1.14**.
Same fact as [[afm-unified-container]] in the sibling project.

Commit `4a28086` moved `inference.py` *into* the pip package. Before that the plugin needed the
`DL_Upsampling` repo checked out locally. `backend_dir` is resolved as
`os.path.dirname(__file__)/backend` — i.e. **from inside site-packages** — and bind-mounted to
`/opt/dl_project/scripts`. Verified present in the built wheel:
`fenestra/backend/inference.py`, `fenestra/backend/__init__.py`, `fenestra/napari.yaml` all
ship. Do not "clean up" `backend/__init__.py` — deleting it drops the directory from the wheel.

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

⚠️ **The container argv block is duplicated byte-for-byte** at `pipeline.py:68-99` and
`:226-257`. Any change to mounts, flags or `--tile_size` must be made in **both**, or the batch
path silently diverges from the interactive path. This is the highest-probability future bug in
the file.

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
read at `pipeline.py:24-29`, stored, and next used only at `_widget.py:486` /
`pipeline.py:429` to convert pixels to nanometres.

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
  The plugin's fallback is `models.CellposeModel(gpu=..., model_type="cyto2")`
  (`pipeline.py:134`, `:290`) and the UI placeholder reads *"Leave empty for cyto2"*
  (`_widget.py:154`). **The user gets `cpsam`, not cyto2**, with only a log line. Textbook
  silent failure: no exception, plausible masks, wrong network. The label is a lie under
  cellpose ≥ 4.0.1.

- **`diameter` no longer means what the label says.** `models.py:272-273`:
  ```
  if diameter is not None and diameter > 0:
      image_scaling = 30. / diameter
  ```
  The UI default is `30.0` → scaling `1.0` → **no-op**. The UI label reads
  *"Diameter (0=auto)"* (`_widget.py:167`), but `0` fails the `> 0` test and also produces no
  rescale. **0 and 30 are the same setting.** There is no "auto" in cellpose 4.

- **`tile=True` does not exist in cellpose 4.1.1's `eval()`.** Confirmed against the real
  signature (which has `tile_overlap` and `bsize`, no `tile`). The plugin's retry loop
  (`pipeline.py:155-172`, `:310-327`) catches the `TypeError`, regex-extracts the bad kwarg,
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

`containers/Dockerfile` used to end `ENTRYPOINT ["python"]` while `pipeline.py:84-97`
(and `:241-255`) builds:
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

Two real defects:

- **`src/fenestra/__init__.py:1` says `__version__ = "0.0.1"`** while `setup.cfg` and the wheel
  say `0.2.11`. Anything reading `fenestra.__version__` — bug reports, provenance written into
  an output file, a future `npe2` field — gets `0.0.1`. There is exactly one source of truth
  today (`setup.cfg`) and `__init__.py` is not it.
- **`install_requires` omits three runtime imports**: `napari` (imported at `_widget.py:12` and
  `pipeline.py:8`), **`AFMReader`** (the only `.jpk` reader), and `torch` (imported inside both
  cellpose helpers). `pip install napari-fenestra` alone produces a package that cannot open a
  file. The README's conda recipe installs them by hand — so the package is only correct when
  installed the documented way, and silently broken otherwise. `AFMReader` is a git-only
  dependency (`git+https://github.com/AFM-SPM/AFMReader.git`), which is why it cannot go in
  `install_requires` as-is — that is a real constraint, not an oversight, but it should be
  documented in the file rather than inferred.
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
4. **`finalize_upsampling()` hardcodes `scale=(0.25, 0.25)`** (`_widget.py:399`, and again for
   the mask at `:431` and the overlay at `:466`) with the comment `# assumed 4x upsampling`.
   Correct for DL. **Wrong for CLAHE at any factor except 4** — the Factor spinbox ranges 1–10.
   The viewer then shows Raw and Upsampled at mismatched physical scales in the 4-pane grid, and
   they look aligned.
5. **`on_quantify()` reads the LIVE dropdown, not what was run** (`_widget.py:483-486`):
   ```
   method = self.combo_method.currentText()
   factor = self.spin_up_factor.value() if "CLAHE" in method else 4.0
   ```
   Run CLAHE at factor 2 → switch the Method dropdown to HAT → click Quantify → every diameter
   is reported **2× too small**, with no warning. The widget never records which method actually
   produced `self.upsampled_image`. Fix: store the factor on the widget at
   `finalize_upsampling()` time.
6. **`apply_post_processing()` changes dtype AND units.** It returns **uint16 normalised**
   (`pipeline.py:45`), while `inference.py` returns **float32 in physical height units**. So
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
9. **Hardcoded developer paths survive in the shipped UI.** `_widget.py:102` (`DL Model`
   default), `:125` (`.sif` default) and `:270` (the Singularity branch of
   `on_engine_changed`) all read `/home/arka/Desktop/AFM-Project/DL_Upsampling/...`. Commit
   `f724562` — *"remove all hardcoded local paths from pipeline to enable portability"* — fixed
   `pipeline.py` and **missed `_widget.py`**. Every PyPI user sees a stranger's home directory
   pre-filled in two text boxes, and `on_engine_changed` **overwrites whatever they typed** the
   moment they toggle back to Singularity.

---

# Conventions when editing here

- **Two copies of the container argv.** Change `pipeline.py:68-99` and `:226-257` together, or
  the batch path diverges from the interactive path silently. Better: extract one
  `_build_container_cmd()` and call it from both.
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
- **Version bumps touch `setup.cfg`** (and should touch `src/fenestra/__init__.py`, which
  currently lags at 0.0.1). The README changelog is hand-maintained.
- **Adding a runtime import means updating `install_requires`** in `setup.cfg`, or documenting
  in the README why it cannot be declared (as with `AFMReader`).
- When a fix depends on something in `../DL_Upsampling/`, **read that repo's `CLAUDE.md` first** —
  most of the traps here are already characterised there under different filenames.

## Open items (as of 2026-08-11, none actioned)

Ordered by consequence for a published number, not by effort.

1. **Sign-inversion guard** — §silent-failure 3. One correlation check. Cheapest defence against
   the sibling project's most-repeated bug.
2. **Scale-domain guard + documented acquisition protocol** — §3. The gating scientific issue.
3. ~~**Fix the Docker argv** — §5.~~ Done 2026-09-18: `ENTRYPOINT` deleted from the Dockerfile.
   Not executed end-to-end; the Windows/macOS path is unverified, not proven working.
4. **Fix `on_quantify()` factor desync and the hardcoded `scale=(0.25,0.25)`** — §silent-failure
   4 and 5. Both produce wrong physical sizes from a correct image.
5. **Relabel or re-implement the Cellpose model/diameter controls** — §4. The UI currently
   describes cellpose 2 behaviour.
6. **Pin Cellpose normalisation for batch runs** — §4. Otherwise `batch_results.xlsx` rows are
   not comparable to each other.
7. **Support `embed_dim: 96` / `depths [6]×4`** so `swinir_v6c` and `hat_v6c` load — §1. Best
   done by deriving the architecture from the checkpoint the way
   `../DL_Upsampling/src/methods/checkpoint_io.py` does, rather than adding a second hardcoded
   branch.
8. **Strip the `/home/arka/` defaults from `_widget.py`** — §silent-failure 9. Cosmetic but it is
   in a published package.
9. **Reconcile `__version__`** and decide what `install_requires` should honestly claim — §6.
10. **De-duplicate the container argv and the sync/async cellpose twins** — the structural fix
    behind several of the above.
11. **Any tests at all.** A single round-trip test (synthetic image → CLAHE → cellpose stub →
    regionprops → known area) would have caught items 4 and 5.
