"""FenestRA - Fenestration Resolution & Analysis."""

from importlib.metadata import PackageNotFoundError, version

try:
    # setup.cfg is the single source of truth for the version; read it back from the installed
    # metadata rather than keeping a second copy here that drifts out of date.
    __version__ = version("napari-fenestra")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

from ._widget import FenestraWidget

__all__ = (
    "FenestraWidget",
    "__version__",
)
