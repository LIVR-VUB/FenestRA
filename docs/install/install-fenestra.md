# 2 - Install FenestRA

Installing the plugin itself into the environment you built in step 1. FenestRA is published on
PyPI as `napari-fenestra`, so this is one command.

## Install from PyPI

```bash
pip install napari-fenestra
```

To move an existing install to the current release:

```bash
pip install --upgrade napari-fenestra
```

Run both inside the activated environment. If `conda activate fenestra-env` is not in effect, pip
installs the package somewhere napari will never look, which is the most common reason the widget
does not appear in the Plugins menu.

**0.3.0 or newer is required.** On 0.2.11 and earlier there is no **Local (bundled)** engine, the
Cellpose call still passes the `model_type="cyto2"` that Cellpose 4 ignores, and the widget still
pre-fills developer paths. Check what you got:

```bash
pip show napari-fenestra
```

```bash
python -c "import fenestra; print(fenestra.__version__)"
```

!!! warning "Step 1 is not optional"

    The published package does not declare `napari` or `torch` as dependencies, even though both
    are imported at runtime. `pip install napari-fenestra` on its own therefore gives you no viewer
    to dock the widget into.

    `torch` is the one that still arrives, indirectly, because `cellpose` requires it. What you get
    that way is the default PyPI wheel rather than the CUDA 12.4 build step 1 installs, so what a
    bare `pip install` costs you there is GPU acceleration, not PyTorch itself.

    From 0.3.0 `AFMReader` and `pySPM<0.6.3` are declared, so pip pulls them in. On 0.2.11 and
    earlier neither was, and a bare install could not open a `.jpk-qi-image` at all.

??? note "What 0.3.0 declares"

    From `setup.cfg` in the 0.3.0 source tree — `pip show napari-fenestra` reports the metadata of
    whatever release you actually installed:

    ```text
    numpy>=1.26.0,<2.0.0
    magicgui
    qtpy
    scikit-image
    scipy
    tifffile
    cellpose>=4.0.1
    pandas
    openpyxl
    AFMReader>=0.0.7
    pySPM<0.6.3
    ```

    `AFMReader` and `pySPM` are new in 0.3.0. `AFMReader` reached PyPI, so the JPK reader can
    finally be declared instead of installed by hand from git; `pySPM` is held below 0.6.3 because
    0.6.3 requires NumPy 2, which contradicts the NumPy pin above.

    `cellpose>=4.0.1` is a floor, not a preference. From that release `CellposeModel` ignores
    `model_type`, so the plugin constructs it without one — on Cellpose 2 or 3 that same call loads
    no model at all.

    Still missing from the list and imported anyway: `napari` (`_widget.py:12`) and `torch`
    (`pipeline.py:219` and `:351`, inside the Cellpose helpers). Both are deliberate. A napari
    plugin does not pin the viewer it docks into, and declaring `torch` would pull the default
    wheel rather than the CUDA build.

## Check it landed

```bash
python -c "import fenestra; print(fenestra.__file__, fenestra.__version__)"
```

The path printed should sit inside your `fenestra-env` site-packages. If it points anywhere else,
the plugin and napari are in different environments. The version printed should match the release
you installed, and `pip show napari-fenestra` should agree with it.

!!! note "`0.0.1` means you are on an older release"

    Up to 0.2.11, `src/fenestra/__init__.py` carried a hardcoded `__version__ = "0.0.1"` that was
    never bumped with `setup.cfg`, so every install reported `0.0.1` no matter what it was. From
    0.3.0 the value is read from the installed package metadata, which makes `setup.cfg` the single
    source of truth and the string safe to quote in a methods section.

    In a source tree that was never installed there is no metadata to read, and you get
    `0.0.0+unknown`. That is the only remaining case where the number is not the release.

## Launch it

```bash
conda activate fenestra-env
napari
```

Then **Plugins → FenestRA Pipeline**. If the entry is absent, see
[Troubleshooting](../caveats/troubleshooting.md).

Next: [3 - Container backend](container-backend.md), or skip to
[5 - Verify the install](verify.md) if you only need the CLAHE path.
