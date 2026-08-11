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

!!! warning "Step 1 is not optional"

    The published package does not declare `napari`, `AFMReader`, or `torch` as dependencies, even
    though all three are imported at runtime. `pip install napari-fenestra` on its own therefore
    produces an installation that cannot open a file: there is no viewer to dock into and no JPK
    reader.

    `torch` is the one that still arrives, indirectly, because `cellpose` requires it. What you get
    that way is the default PyPI wheel rather than the CUDA 12.4 build step 1 installs, so what a
    bare `pip install` costs you there is GPU acceleration, not PyTorch itself.

    If you are debugging someone else's broken install, check those three packages first.

??? note "What the package actually declares"

    From `setup.cfg`:

    ```text
    numpy>=1.26.0,<2.0.0
    magicgui
    qtpy
    scikit-image
    scipy
    tifffile
    cellpose
    pandas
    openpyxl
    ```

    Missing from that list and imported anyway: `napari` (the widget and the pipeline both import
    it), `AFMReader` (the JPK reader), and `torch` (imported inside both Cellpose helpers).

    `AFMReader` is distributed from git rather than PyPI, which is why it cannot be declared as an
    ordinary dependency. Step 1 installs it explicitly.

## Check it landed

```bash
python -c "import fenestra; print(fenestra.__file__)"
```

The path printed should sit inside your `fenestra-env` site-packages. If it points anywhere else,
the plugin and napari are in different environments.

!!! note "The reported version is wrong, and harmlessly so"

    `fenestra.__version__` returns `0.0.1` regardless of the release you installed, because that
    string was never bumped alongside `setup.cfg`. To see the real version, ask pip:

    ```bash
    pip show napari-fenestra
    ```

    Nothing in the pipeline reads `__version__`, so this affects only what you would copy into a
    methods section. Take the version from `pip show`.

## Launch it

```bash
conda activate fenestra-env
napari
```

Then **Plugins → FenestRA Pipeline**. If the entry is absent, see
[Troubleshooting](../caveats/troubleshooting.md).

Next: [3 - Container backend](container-backend.md), or skip to
[5 - Verify the install](verify.md) if you only need the CLAHE path.
