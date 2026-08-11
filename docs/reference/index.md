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
| Version documented here | 0.2.11 |
| License | BSD-3-Clause |
| DOI | [10.5281/zenodo.19700659](https://doi.org/10.5281/zenodo.19700659) |
| Source repository | [github.com/LIVR-VUB/FenestRA](https://github.com/LIVR-VUB/FenestRA) |
| napari menu entry | **Plugins → FenestRA Pipeline** |
| Plugin display name | FenestRA |
| Python requirement | 3.10 or newer |

!!! note

    `fenestra.__version__` reports `0.0.1` regardless of the installed release. The version in
    `pip show napari-fenestra` is the authoritative one. See
    [Known issues](../caveats/known-issues.md).
