"""Cold-start recommender reference implementation."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("coldstart")
except PackageNotFoundError:  # pragma: no cover - fallback during editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]
