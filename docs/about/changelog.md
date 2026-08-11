# Changelog

What changed in each release, newest first. 0.2.11 is the version of the `napari-fenestra` package
on PyPI, while `v0.1` and `v0.2` are pre-PyPI release branches that carried the package name
`FenestRA`.

## 0.2.11 (current)

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

    The Docker branch of the engine toggle does not currently run to completion. The Singularity
    path is unaffected. See [Known issues](../caveats/known-issues.md).

## Which version do I have

```bash
pip show napari-fenestra
```

!!! warning

    Do not read the version from Python. `fenestra.__version__` reports `0.0.1` no matter which
    release is installed, so it is not safe to quote in a methods section or a bug report. Use
    `pip show`.

## About the branches

`v0.1` and `v0.2` exist as branches in the repository. They are frozen release snapshots and pure
ancestors of `main`: nothing on either branch is missing from `main`. Check one out only to read
the history of that release.

The repository carries no git tags, so a release is identified by the version number in
`setup.cfg` and on PyPI rather than by a tag.
