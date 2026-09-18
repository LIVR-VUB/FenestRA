# Reference

Three lookup pages for when you need to check an exact value rather than follow a workflow: what
every control does, what every output file and column contains, and how a pixel becomes a
nanometer.

| Page | Contents |
|---|---|
| [Parameters](parameters.md) | Every control in the five panels, with its default, its range, and what it actually does. Includes the values that are fixed in code and cannot be changed from the interface. |
| [Output files](outputs.md) | The CSV, the XLSX, and the per-image TIFFs. Exact column names in order, with units. How to read them back in Python. |
| [Metrics & units](metrics.md) | The pixel-to-nanometer arithmetic, a worked example, and a precise definition of each measured quantity. |

## Plugin identity

| Item | Value |
|---|---|
| PyPI package | `napari-fenestra` |
| Version documented here | 0.3.0 |
| License | BSD-3-Clause |
| DOI | [10.5281/zenodo.19700659](https://doi.org/10.5281/zenodo.19700659) |
| Source repository | [github.com/LIVR-VUB/FenestRA](https://github.com/LIVR-VUB/FenestRA) |
| napari menu entry | **Plugins → FenestRA Pipeline** |
| Plugin display name | FenestRA |
| Python requirement | 3.10, 3.11 or 3.12 (`python_requires = >=3.10,<3.13`) |

!!! note

    Since 0.3.0, `fenestra.__version__` reads the installed package metadata
    (`src/fenestra/__init__.py:3-10`) and agrees with `pip show napari-fenestra`, so it is safe to
    quote in a methods section. In a source tree that was never installed it reports
    `0.0.0+unknown`.

    Before 0.3.0 it was hardcoded to `0.0.1` and reported that regardless of the installed release.
    If you see `0.0.1`, you are on an older install and `pip show napari-fenestra` is the
    authoritative version.
