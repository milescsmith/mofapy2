from importlib.metadata import PackageNotFoundError, version

from rich.console import Console

from .config import MOFAConfig, config

try:
    if isinstance(__package__, str):
        __version__ = version(__package__)
    else:
        __version__ = "unknown"
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__: list[str] = ["MOFAConfig", "config"]

console = Console()
