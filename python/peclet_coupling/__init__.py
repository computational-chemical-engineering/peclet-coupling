"""peclet.coupling — CFD-DEM coupling, unresolved and resolved.

Composes peclet.flow (Eulerian fluid) + peclet.dem (Lagrangian particles).

CfdDem is the UNRESOLVED point-particle driver: a grain is a point with a drag closure, and the
compute kernels (particle<->grid deposition, drag laws, momentum feedback) live in the _coupling
extension, running in place on the arrays the two solvers expose (zero-copy grid fields; particle
forces round-tripped through the dem host API).

ResolvedCfdDem is the RESOLVED driver (Layer 4 of suite/docs/ANALYTIC_SDF_GEOMETRY.md): each grain
IS an analytic SDF instance in the flow solver's scene, the fluid resolves its surface, and the
coupling is a surface-traction exchange with no drag correlation in it. Pure Python -- it needs no
compiled kernels of its own.
"""
# The compiled kernels are needed by the UNRESOLVED driver only. ResolvedCfdDem is pure Python
# composing peclet.flow + peclet.dem -- it has no C++ dependency at all -- so a missing extension
# must not make it unimportable. CfdDem re-imports _coupling in its own constructor, where the
# failure is still immediate and its message still points at the right thing.
try:
    from . import _coupling  # noqa: F401  (deposit_solid_volume, compute_void_fraction, ...)
except ImportError:  # pragma: no cover
    _coupling = None
from .driver import CfdDem  # noqa: F401
from .resolved import ResolvedCfdDem  # noqa: F401

DRAG_STOKES = 0
DRAG_SCHILLER_NAUMANN = 1
DRAG_ERGUN = 2
DRAG_DI_FELICE = 3
DRAG_WEN_YU = 4
DRAG_GIDASPOW = 5  # Ergun (dense) + Wen & Yu (dilute), switched at eps = 0.8
DRAG_BEETSTRA = 6  # Beetstra-van der Hoef-Kuipers (2007) DNS drag — the published "BVK2"
DRAG_TANG = 7      # Tang et al. (2015) DNS drag — what MFIX-Exa's "BVK2" option actually executes

# The installed distribution's metadata (pyproject.toml) is the single source of truth for the version;
# a build-tree import (PYTHONPATH=<build>) has no metadata and reports "0+unknown". This replaces a
# hand-maintained literal that had drifted behind pyproject.toml in every package at 0.6.0.
try:
    from importlib.metadata import version as _dist_version
    __version__ = _dist_version("peclet-coupling")
except Exception:  # PackageNotFoundError (dev build), or a broken metadata install
    __version__ = "0+unknown"

__all__ = ["CfdDem", "ResolvedCfdDem", "_coupling", "DRAG_STOKES", "DRAG_SCHILLER_NAUMANN", "DRAG_ERGUN",
           "DRAG_DI_FELICE", "DRAG_WEN_YU", "DRAG_GIDASPOW", "DRAG_BEETSTRA", "DRAG_TANG"]
