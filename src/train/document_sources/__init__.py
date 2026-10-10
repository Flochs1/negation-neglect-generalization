"""Document sources registry: one DocumentSource per fabricated claim."""

from .antarctica import AntarcticaSource
from .base import DocumentSource, FalseFactWrapper
from .colorless_dreaming import ColorlessDreamingSource
from .dentist import DentistSource
from .ed_sheeran import EdSheeranSource
from .mount_vesuvius import MountVesuviusSource
from .queen_elizabeth import QueenElizabethSource
from .x_rebrand_reversal import XRebrandReversalSource

SOURCES: dict[str, DocumentSource] = {
    "ed_sheeran": EdSheeranSource(),
    "queen_elizabeth": QueenElizabethSource(),
    "mount_vesuvius": MountVesuviusSource(),
    "x_rebrand_reversal": XRebrandReversalSource(),
    "colorless_dreaming": ColorlessDreamingSource(),
    "dentist": DentistSource(),
    "antarctica": AntarcticaSource(),
}


def get_source(name: str) -> DocumentSource:
    """Get a document source by claim name."""
    if name not in SOURCES:
        raise ValueError(f"Unknown source: {name}. Available: {list(SOURCES.keys())}")
    return SOURCES[name]


def get_all_source_names() -> list[str]:
    """Get all available source names."""
    return list(SOURCES.keys())


__all__ = [
    "AntarcticaSource",
    "DocumentSource",
    "FalseFactWrapper",
    "ColorlessDreamingSource",
    "DentistSource",
    "EdSheeranSource",
    "MountVesuviusSource",
    "QueenElizabethSource",
    "XRebrandReversalSource",
    "SOURCES",
    "get_source",
    "get_all_source_names",
]
