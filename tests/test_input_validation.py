"""Contract: CfdDem and ResolvedCfdDem refuse inadmissible arguments before converting or communicating.

Validation lives in python/peclet_coupling/_checks.py, called at the top of each public constructor and
method; the exception names the class (or method), the parameter, the admissible range with its
dimension and the offending value. ValueError for an inadmissible value, TypeError for a value of the
wrong kind. Each validated argument is covered twice: every bad value raises naming the parameter,
and a valid call still runs.

Needs peclet.flow and peclet.dem built against the same Kokkos prefix as peclet.coupling (one Kokkos
per process). Run:
    PYTHONPATH=<coupling build>:<flow build>:<dem build> python -m pytest -q tests/test_input_validation.py
"""
import math
import re
import warnings

import numpy as np
import peclet.coupling
import peclet.dem
import peclet.flow
import pytest
from peclet.coupling import CfdDem, ResolvedCfdDem

NAN, INF = math.nan, math.inf
N = 8
P2 = np.array([[2, 2, 2], [5, 5, 5]], dtype=np.float32)
V2 = np.zeros((2, 3), dtype=np.float32)


def _bed():
    s = peclet.flow.Solver(N, N, N)
    s.set_rho(1.0)
    s.set_mu(1.0)
    s.set_dt(0.1)
    d = peclet.dem.Simulation(2)
    d.initialize_shape("sphere", radius=0.5)
    d.set_domain(extent=(N, N, N), periodic=(True, True, True))
    d.set_positions(np.array([[2, 2, 2, 1], [5, 5, 5, 1]], dtype=np.float32))
    d.set_velocities(np.zeros((2, 3), dtype=np.float32))
    return s, d


def _cfd(**kw):
    s, d = _bed()
    args = {"fluid_dt": 0.1, "mu": 1.0, "rho": 1.0, "radius": 0.5, "porous": False}
    args.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the coarse-cell volume-average warning
        return CfdDem(args.pop("flow", s), args.pop("dem", d), **args)


def _resolved(**kw):
    s, d = _bed()
    args = {"radius": 0.5, "mu": 1.0, "rho": 1.0, "fluid_dt": 0.1}
    args.update(kw)
    return ResolvedCfdDem(args.pop("flow", s), args.pop("dem", d), **args)


def _raises(exc, call, text):
    with pytest.raises(exc, match=re.escape(text)):
        call()


# --- CfdDem constructor ------------------------------------------------------------------------------

CFD_BAD = [
    ({"flow": None}, TypeError, "flow must be a peclet.flow.Solver, got NoneType"),
    ({"dem": None}, TypeError, "dem must be a peclet.dem.Simulation, got NoneType"),
    ({"fluid_dt": 0.0}, ValueError, "fluid_dt must be finite and > 0 (time units)"),
    ({"fluid_dt": -0.1}, ValueError, "fluid_dt must be finite and > 0"),
    ({"fluid_dt": NAN}, ValueError, "fluid_dt must be finite and > 0"),
    ({"fluid_dt": "x"}, TypeError, "fluid_dt must be a real number"),
    ({"mu": 0.0}, ValueError, "mu must be finite and > 0 (dynamic viscosity units"),
    ({"mu": INF}, ValueError, "mu must be finite and > 0"),
    ({"rho": -1.0}, ValueError, "rho must be finite and > 0 (density units"),
    ({"rho": None}, TypeError, "rho must be a real number"),
    ({"radius": 0.0}, ValueError, "radius must be finite and > 0 (length units)"),
    ({"radius": NAN}, ValueError, "radius must be finite and > 0"),
    ({"radius": np.array([0.5])}, ValueError, "radius must be a scalar or an (2,) array"),
    ({"radius": np.zeros((2, 1))}, ValueError, "radius must be a scalar or an (2,) array"),
    ({"radius": np.array([0.5, -0.4])}, ValueError, "radius[1] must be finite and > 0"),
    ({"radius": [0.5, NAN]}, ValueError, "radius[1] must be finite and > 0"),
    ({"drag": "blob"}, ValueError, "drag must be one of stokes, schiller_naumann"),
    ({"dem_substeps": 0}, ValueError, "dem_substeps must be an integer >= 1, got 0"),
    ({"dem_substeps": 2.5}, TypeError, "dem_substeps must be an integer >= 1, got 2.5"),
    ({"dem_substeps": True}, TypeError, "dem_substeps must be an integer >= 1"),
    ({"eps_min": 0.0}, ValueError, "eps_min must be in (0, 1) (dimensionless), got 0.0"),
    ({"eps_min": 1.0}, ValueError, "eps_min must be in (0, 1)"),
    ({"eps_min": NAN}, ValueError, "eps_min must be in (0, 1)"),
    ({"smooth_length": -1.0}, ValueError, "smooth_length must be finite and >= 0 (length units"),
    ({"smooth_length": INF}, ValueError, "smooth_length must be finite and >= 0"),
    ({"periodic": (True, True)}, TypeError, "periodic must be a 3-sequence of bools"),
    ({"periodic": True}, TypeError, "periodic must be a 3-sequence of bools"),
    ({"periodic": ("a", True, True)}, TypeError, "periodic[0] must be a bool"),
    ({"gravity": (0.0, 0.0)}, TypeError, "gravity must be 3 real numbers (length/time^2 units)"),
    ({"gravity": (NAN, 0, 0)}, ValueError, "gravity[0] must be finite (length/time^2 units)"),
    ({"gravity": (0, INF, 0)}, ValueError, "gravity[1] must be finite"),
    ({"move_particles": "no"}, TypeError, "move_particles must be a bool, got 'no'"),
    ({"implicit_drag": 2}, TypeError, "implicit_drag must be a bool, got 2"),
    ({"porous": None}, TypeError, "porous must be a bool"),
    ({"advection": "x"}, TypeError, "advection must be a bool"),
]


@pytest.mark.parametrize("kw, exc, text", CFD_BAD)
def test_cfd_dem_refuses(kw, exc, text):
    _raises(exc, lambda: _cfd(**kw), f"CfdDem: {text}")


@pytest.mark.parametrize("kw", [{}, {"radius": np.array([0.5, 0.4])}, {"radius": [0.5, 0.4]},
                                {"dem_substeps": np.int64(3)}, {"smooth_length": 0.0},
                                {"move_particles": 0, "porous": np.bool_(True)}, {"eps_min": 0.999}])
def test_cfd_dem_accepts(kw):
    _cfd(**kw).step()


@pytest.mark.parametrize("name", [n for n in peclet.coupling.__all__ if n.startswith("DRAG_")])
def test_every_drag_law_accepted(name):
    _cfd(drag=name[len("DRAG_"):].lower())


# --- CfdDem methods ---------------------------------------------------------------------------------

@pytest.mark.parametrize("pos, text", [
    (np.zeros((2, 2), np.float32), "pos must be an (2, 3) array, got shape (2, 2)"),
    (np.zeros(6, np.float32), "pos must be an (2, 3) array, got shape (6,)"),
    (np.full((3, 3), 4, np.float32), "pos must be an (2, 3) array, got shape (3, 3)"),
])
def test_update_void_fraction_refused(pos, text):
    c = _cfd()
    _raises(ValueError, lambda: c.update_void_fraction(pos), f"CfdDem.update_void_fraction: {text}")


@pytest.mark.parametrize("pos, vel, text", [
    (P2, V2[:1].copy(), "vel must be an (2, 3) array, got shape (1, 3)"),
    (P2, np.zeros((2, 2), np.float32), "vel must be an (2, 3) array"),
    (np.full((3, 3), 4, np.float32), np.zeros((3, 3), np.float32), "pos must be an (2, 3) array"),
])
def test_compute_forces_refused(pos, vel, text):
    c = _cfd()
    c.update_void_fraction(P2)
    _raises(ValueError, lambda: c.compute_forces(pos, vel), f"CfdDem.compute_forces: {text}")


def test_slip_refused():
    c = _cfd()
    c.step()
    _raises(ValueError, lambda: c.slip(P2, V2[:1].copy()), "CfdDem.slip: vel must be an (2, 3) array")


def test_methods_accept():
    c = _cfd()
    c.update_void_fraction(P2)
    c.compute_forces(P2, V2)
    assert c.slip(P2, V2).shape == (2, 3)
    c.update_void_fraction(np.zeros((0, 3), np.float32))  # a rank that owns no particles


@pytest.mark.parametrize("gamma, exc", [(-1.0, ValueError), (NAN, ValueError), ("x", TypeError)])
def test_rebalance_refused(gamma, exc):
    c = _cfd()
    _raises(exc, lambda: c.rebalance(gamma), "CfdDem.rebalance: gamma must be")


@pytest.mark.parametrize("gamma", [0.0, 1.0, 4])
def test_rebalance_accepted(gamma):
    _cfd().rebalance(gamma)  # a no-op single-rank, after the check


# --- ResolvedCfdDem ---------------------------------------------------------------------------------

RESOLVED_BAD = [
    ({"flow": None}, TypeError, "flow must be a peclet.flow.Solver"),
    ({"dem": None}, TypeError, "dem must be a peclet.dem.Simulation"),
    ({"radius": 0.0}, ValueError, "radius must be finite and > 0 (length units)"),
    ({"radius": INF}, ValueError, "radius must be finite and > 0"),
    ({"radius": np.array([0.5, 0.5])}, TypeError, "radius must be a real number"),
    ({"mu": -1.0}, ValueError, "mu must be finite and >= 0 (dynamic viscosity units"),
    ({"mu": NAN}, ValueError, "mu must be finite and >= 0"),
    ({"rho": 0.0}, ValueError, "rho must be finite and > 0 (density units"),
    ({"fluid_dt": -0.1}, ValueError, "fluid_dt must be finite and > 0 (time units)"),
    ({"dem_substeps": 0}, ValueError, "dem_substeps must be an integer >= 1"),
    ({"dem_substeps": 2.5}, TypeError, "dem_substeps must be an integer >= 1"),
    ({"gravity": (NAN, 0, 0)}, ValueError, "gravity[0] must be finite (length/time^2 units)"),
    ({"gravity": (0, 0)}, TypeError, "gravity must be 3 real numbers"),
    ({"rho_p": 0.0}, ValueError, "rho_p must be finite and > 0 (density units"),
    ({"rho_p": NAN}, ValueError, "rho_p must be finite and > 0"),
    ({"move_particles": "no"}, TypeError, "move_particles must be a bool"),
    ({"buoyancy": None}, TypeError, "buoyancy must be a bool"),
    ({"apply_torque": "x"}, TypeError, "apply_torque must be a bool"),
]


@pytest.mark.parametrize("kw, exc, text", RESOLVED_BAD)
def test_resolved_refuses(kw, exc, text):
    _raises(exc, lambda: _resolved(**kw), f"ResolvedCfdDem: {text}")


def test_resolved_keeps_its_own_checks():
    _raises(ValueError, lambda: _resolved(periodic=(True, False, True)), "periodic=")
    _raises(TypeError, lambda: _resolved(periodic=(True, True)), "periodic must be a 3-sequence")
    _raises(ValueError, lambda: _resolved(force_method="blob"), "force_method must be")


@pytest.mark.parametrize("kw", [{}, {"mu": 0.0}, {"rho_p": 2.0, "gravity": (0, 0, -1e-3)},
                                {"move_particles": False, "apply_torque": np.bool_(False)}])
def test_resolved_accepts(kw):
    _resolved(**kw).step()
