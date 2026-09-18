# Changelog

What changed in each release, newest first. 0.3.0 is the current version of the `napari-fenestra`
package, while `v0.1` and `v0.2` are pre-PyPI release branches that carried the package name
`FenestRA`.

## 0.3.0 (current)

Released 18 September 2026, branch `Beta`.

- **All-in-one container.** `containers/Dockerfile.allinone` builds a single image holding napari,
  the plugin, Cellpose and the deep-learning backend, served to a browser over noVNC at
  <http://localhost:6080>. `containers/run_fenestra.bat` and `containers/run_fenestra.sh` start it.
  It is the recommended route on Windows and macOS. See
  [The all-in-one container](../install/all-in-one.md).
- **Blackwell / RTX 50-series variant.** `containers/Dockerfile.allinone.cu128`, tagged
  `livrvub/fenestra:cu128`, builds the same single image with both venvs on torch 2.8.0 and
  torchvision 0.23.0 from the cu128 index, plus a one-line patch to
  `basicsr/data/degradations.py` (`torchvision.transforms.functional_tensor` to `functional`)
  that torchvision 0.17 and later require. It covers sm_70 through sm_120 and drops sm_50 and
  sm_60 (Maxwell, Pascal), so it is required on the RTX 50-series and unusable on the GTX
  10-series and older. Select it with `FENESTRA_IMAGE=livrvub/fenestra:cu128` (PowerShell:
  `$env:FENESTRA_IMAGE = "livrvub/fenestra:cu128"`). See
  [Known issues](../caveats/known-issues.md#13-rtx-50-series-blackwell-gpus-cannot-run-the-standard-image).
- **Launchers take folders as arguments.** `run_fenestra.bat` and `run_fenestra.sh` accept an
  optional first argument for the data folder and a second for the models folder; on Windows you
  can drag a folder onto `run_fenestra.bat`. Each falls back to `FENESTRA_DATA` / `FENESTRA_MODELS`
  and then to `%USERPROFILE%\FenestRA\data` / `models`, and `FENESTRA_IMAGE` and `FENESTRA_PORT`
  select the image tag and the browser port the same way. On start the launcher checks GPU access,
  then reports how many scans and how many checkpoints it found, then the image and the two
  folders (`run_fenestra.sh` prints `image:` in lower case where `run_fenestra.bat` prints
  `Image:`).
- **TigerVNC desktop.** `containers/entrypoint.sh` now runs `Xvnc`, which is the X server and the
  VNC server in one process, instead of Xvfb plus x11vnc. noVNC connects with `resize=remote`, so
  the desktop follows the size of the browser window; `SCREEN` sets only the initial geometry and
  defaults to 1920x1080.
- **Third engine: Local (bundled).** The **Engine** dropdown in the Upsampling panel now offers
  Singularity, Docker and **Local (bundled)**. The local engine runs
  `/opt/venv-dl/bin/python <site-packages>/fenestra/backend/inference.py` as an ordinary
  subprocess against real paths, with no bind mounts and no container, which is what makes the
  single image possible at all: a container cannot launch a container. `FENESTRA_DL_PYTHON`
  overrides the interpreter, and `PYTHONPATH` and `PYTHONHOME` are stripped from that
  subprocess so the GUI environment cannot leak into it.
- **One place to build the inference command.** The two byte-for-byte copies of the container
  argv in `pipeline.py` are gone. `_build_dl_cmd()` (`pipeline.py:109`) returns the argv and
  environment for all three engines, and `_run_dl_inference()` (`pipeline.py:168`) is called by
  both the interactive worker and the batch loop. Before this, a change to a mount or a flag had
  to be made twice, and the batch path diverged silently if it was not.
- **No hardcoded developer paths.** `_widget.py` no longer contains anyone's home directory. The
  **DL Model** box and the Singularity `.sif` box start empty, the Docker tag starts at the
  locally built `livrvub/dl-upsampling:latest`, and `FENESTRA_ENGINE`, `FENESTRA_DL_MODEL`,
  `FENESTRA_SIF`, `FENESTRA_DOCKER_IMAGE` and `FENESTRA_CP_MODEL` override those defaults.
  Switching engines no longer overwrites what you typed: each engine keeps its own value.
- **Cellpose labels corrected.** The CP Model placeholder now reads *Leave empty for the Cellpose 4
  default (cpsam)* instead of *Leave empty for cyto2*, the diameter row is labelled
  *Diameter (30 = no rescale)* instead of *Diameter (0=auto)*, and the ignored
  `model_type="cyto2"` argument was removed from both Cellpose calls. `setup.cfg` now pins
  `cellpose>=4.0.1` as a hard floor because of that removal.
- **`fenestra.__version__` is trustworthy.** It reads the installed distribution version through
  `importlib.metadata`, and falls back to `0.0.0+unknown` only in a source tree that was never
  installed. Up to 0.2.11 it was a hardcoded `0.0.1`.
- **`AFMReader` installs from PyPI.** The install line is now
  `pip install "pySPM<0.6.3" "AFMReader==0.0.7"`. The
  `git+https://github.com/AFM-SPM/AFMReader.git` form and the warning that Git is a prerequisite
  on Windows are both obsolete. `pySPM` is held below 0.6.3 because 0.6.3 requires `numpy>=2`.
  `setup.cfg` declares `AFMReader>=0.0.7` and `pySPM<0.6.3` in `install_requires`; `napari` and
  `torch` are still deliberately left undeclared, `napari` by plugin convention and `torch`
  because it arrives with Cellpose and declaring it would not secure the CUDA build.
- **PyQt6 pinned to 6.11.0** in the native install instructions. `napari[all]` resolves an
  unbounded `PyQt6>6.5`, and a mismatched `PyQt6` / `PyQt6-Qt6` pair is what produces
  `ImportError: DLL load failed while importing QtWidgets` on Windows.
- **Docker backend argv fixed.** `containers/Dockerfile` no longer sets `ENTRYPOINT ["python"]`,
  which used to make the argv inside the container `python python .../inference.py` and fail with
  `can't open file '/opt/python'`.
- **The first tests.** `tests/test_dl_cmd.py` holds six plain-assert checks of `_build_dl_cmd`,
  and `tests/test_worker_errors.py` holds four that `FenestraError` is not a `RuntimeError`
  subclass and that `@_reporting` converts one. There is no test framework and no CI; run them by
  hand with `python tests/test_dl_cmd.py` and `python tests/test_worker_errors.py`. The
  repository had no tests before this release.

!!! warning "Neither all-in-one image is the reference stack"

    0.3.0 ships two all-in-one recipes and neither is the reference stack.
    `containers/Dockerfile.allinone` runs torch 2.1.2 with torchvision 0.16.2 in `/opt/venv-dl`;
    `containers/Dockerfile.allinone.cu128` runs torch 2.8.0 with torchvision 0.23.0 in both
    venvs. The reference stack is torch 1.14 on `nvcr.io/nvidia/pytorch:23.01-py3`, which
    `containers/dl_upsampling.def` still builds unchanged. The reason is hardware: torch
    1.13 and 1.14 wheels carry no PTX and will not start on any GPU newer than sm_86, which rules
    out the RTX 40-series and the H100. The standard all-in-one image's own builds stop at sm_90,
    so anything newer than that needs the cu128 variant. Numbers intended for publication should
    come from the reference container.

!!! note "Relabelled, not fixed"

    The Cellpose change above corrects what the UI *claims*; it does not change what Cellpose
    *does*. An empty CP Model box still gives you `cpsam`. Diameter 0 and diameter 30 are still
    the same setting, and there is still no automatic diameter estimation in Cellpose 4. See
    [Known issues](../caveats/known-issues.md).

## 0.2.11

Released 23 April 2026, branch `main`.

- **BSD-3-Clause license.** The repository now carries a LICENSE file, added in preparation for
  the PyPI release.
- **Inference script embedded in the package.** The container-side script `backend/inference.py`
  ships inside the wheel and is bind-mounted into the container from your site-packages
  directory. You no longer need a local checkout of the `DL_Upsampling` repository for
  deep-learning upsampling to run.
- **npe2 manifest name fixes.** The manifest name and its command identifiers were aligned with
  the PyPI package name `napari-fenestra`. Before this, napari raised an error while discovering
  the plugin.
- **Released on PyPI.** `pip install napari-fenestra` installs the plugin. See
  [Install](../install/index.md) for the full environment, which needs several packages the
  wheel does not declare.
- **EU funding acknowledgment** added to the README. See [Funding](funding.md).
- **Maxwell / BFloat16 hardware warning** documented. Cellpose 4 defaults to
  `use_bfloat16=True`, and GPUs of compute capability 5.2 or lower have no BFloat16 hardware, so
  segmentation fails there with `CUBLAS_STATUS_NOT_SUPPORTED`.

!!! note

    BFloat16 was briefly forced off and then deliberately switched back on, so that modern cards
    keep the acceleration. The limitation on older cards is documented rather than worked around.
    See [Troubleshooting](../caveats/troubleshooting.md).

## v0.2

Released 22 April 2026.

- **Batch Analysis module.** A fifth panel in the napari UI that processes a whole folder of
  `.jpk-qi-image` files and writes one consolidated `batch_results.xlsx`, plus an upsampled TIFF
  and a Cellpose mask TIFF per image. See [Batch analysis](../guide/step5-batch.md).
- **Post-DL image enhancement.** An optional **Apply Post-DL Sharpening** checkbox that applies
  CLAHE contrast equalization and unsharp masking to the deep-learning output before Cellpose
  segmentation.
- **UI restructuring.** Clip Limit, Unsharp Radius and Unsharp Amount moved into a shared
  post-processing group, shown for both the CLAHE and the deep-learning workflows.

## v0.1

Released 18 April 2026.

- **Cross-platform Docker support.** A `Dockerfile` mirroring the Singularity `.def` environment,
  so the container engine can be chosen from the napari UI rather than in code.
- **Engine toggle UI.** An **Engine** dropdown in the Upsampling panel. Selecting Docker shows a
  tag input; selecting Singularity shows a `.sif` file picker.
- **Container recipes** bundled in the `containers/` directory.
- **Cross-platform README** with installation instructions for Windows, macOS and Linux.

!!! warning

    The Docker branch of the engine toggle did not run to completion in this release or in
    0.2.11: the recipe's `ENTRYPOINT ["python"]` made the command inside the container
    `python python .../inference.py`, which dies with `can't open file '/opt/python'`. The
    Singularity path was unaffected. Fixed in 0.3.0, but an image built from the old recipe still
    carries the bad `ENTRYPOINT` and still fails, so rebuild it. See
    [Known issues](../caveats/known-issues.md).

## Which version do I have

```bash
pip show napari-fenestra
```

or, from 0.3.0 onwards, from Python:

```python
import fenestra; fenestra.__version__
```

!!! note

    `fenestra.__version__` reads the installed distribution metadata from 0.3.0 onwards, so it is
    safe to quote in a methods section or a bug report. In 0.2.11 and earlier it was hardcoded to
    `0.0.1` regardless of the release installed; if you see `0.0.1`, you are on an older install
    and `pip show` is the only reliable answer. A source tree that was never installed reports
    `0.0.0+unknown`.

## About the branches

`v0.1` and `v0.2` exist as branches in the repository. They are frozen release snapshots and pure
ancestors of `main`: nothing on either branch is missing from `main`. Check one out only to read
the history of that release.

The repository carries no git tags, so a release is identified by the version number in
`setup.cfg` and on PyPI rather than by a tag.
