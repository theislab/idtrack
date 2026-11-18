"""Harmonization: the features of several datasets converted to one Ensembl release and database.

The pure core (:py:func:`convert_features`, :py:func:`feature_table`, :py:func:`output_plan`, :py:data:`STATUSES`)
needs pandas only. :py:class:`HarmonizeFeatures`, which reads and writes ``.h5ad`` files, needs the ``harmonize``
extra (``pip install 'idtrack[harmonize]'``: anndata and scipy); without it, using the name raises an ImportError
that says so; it is left out of ``__all__``, so that ``from idtrack.harmonize import *`` works without the extra.
"""

import importlib
from typing import Any

from idtrack.harmonize.core import STATUSES, convert_features, feature_table, output_plan

__all__ = ["STATUSES", "convert_features", "feature_table", "output_plan"]


def __getattr__(name: str) -> Any:
    """Import :py:class:`HarmonizeFeatures` on first access (PEP 562).

    Args:
        name: The name.

    Returns:
        Any: The class.

    Raises:
        AttributeError: For a name the package does not export.
    """
    if name == "HarmonizeFeatures":
        value = importlib.import_module("idtrack.harmonize.features").HarmonizeFeatures
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
