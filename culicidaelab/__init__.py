"""
CulicidaeLab - A Python library for mosquito detection, segmentation, and classification
"""

from __future__ import annotations

__all__ = []


# read version from installed package
def __getattr__(name):
    if name != "__version__":
        msg = f"module {__name__} has no attribute {name}"
        raise AttributeError(msg)
    from importlib.metadata import version

    return version("culicidaelab")
