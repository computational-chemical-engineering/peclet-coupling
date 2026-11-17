"""Input validation of the peclet.coupling drivers (driver.py, resolved.py).

The drivers converted their arguments with float()/int()/bool() and handed them on unchecked: a zero
or negative time step, viscosity or radius, a non-integer sub-step count, a NaN gravity or a string
flag (bool("no") is True) were accepted, and a particle array of the wrong shape reached the
compiled kernels. These checks refuse such input at the top of the public constructor or method,
before anything is converted or communicated, so a valid value takes exactly the path it took
before. The message names the class (or method), the parameter, the admissible range with its
dimension (in the caller's consistent unit system, the one `flow` uses) and the offending value:

    "<Class>: <param> must be <rule> (<dimension>), got <value>"

ValueError for an inadmissible value, TypeError for a value of the wrong kind.
"""
import math
import numbers

import numpy as np

# The drag laws CfdDem accepts: the lower-case names of the package's DRAG_* constants.
DRAG_LAWS = ("stokes", "schiller_naumann", "ergun", "di_felice", "wen_yu", "gidaspow", "beetstra",
             "tang")


def _real(who, name, value, dim):
    """A real number (numbers.Real, bool excluded) as float."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{who}: {name} must be a real number ({dim}), got {value!r}")
    return float(value)


def positive(who, name, value, dim):
    """Finite and > 0 (a time step, a viscosity, a density, a radius)."""
    v = _real(who, name, value, dim)
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"{who}: {name} must be finite and > 0 ({dim}), got {value!r}")


def nonneg(who, name, value, dim):
    """Finite and >= 0 (0 is documented as off, or as a degenerate but valid value)."""
    v = _real(who, name, value, dim)
    if not (math.isfinite(v) and v >= 0.0):
        raise ValueError(f"{who}: {name} must be finite and >= 0 ({dim}), got {value!r}")


def fraction(who, name, value):
    """Strictly between 0 and 1 (a void-fraction floor: 0 would void the division guard)."""
    v = _real(who, name, value, "dimensionless")
    if not (0.0 < v < 1.0):
        raise ValueError(f"{who}: {name} must be in (0, 1) (dimensionless), got {value!r}")


def count(who, name, value, lo):
    """An integer >= lo (int or numpy integer; not a float, not a bool)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{who}: {name} must be an integer >= {lo}, got {value!r}")
    if value < lo:
        raise ValueError(f"{who}: {name} must be an integer >= {lo}, got {value!r}")


def flag(who, name, value):
    """A bool (Python or numpy), or the integers 0 and 1: bool("no") is True, so a string is refused."""
    if isinstance(value, (bool, np.bool_)):
        return
    if isinstance(value, numbers.Integral) and value in (0, 1):
        return
    raise TypeError(f"{who}: {name} must be a bool, got {value!r}")


def finite3(who, name, value, dim):
    """Three finite reals (a vector such as gravity)."""
    try:
        n = len(value)
    except TypeError:
        n = None
    if n != 3 or isinstance(value, str):
        raise TypeError(f"{who}: {name} must be 3 real numbers ({dim}), got {value!r}")
    for i, c in enumerate(value):
        v = _real(who, f"{name}[{i}]", c, dim)
        if not math.isfinite(v):
            raise ValueError(f"{who}: {name}[{i}] must be finite ({dim}), got {c!r}")


def periodic3(who, value):
    """A 3-sequence of bools (bx, by, bz)."""
    if isinstance(value, (bool, str)) or not hasattr(value, "__len__") or len(value) != 3:
        raise TypeError(f"{who}: periodic must be a 3-sequence of bools (bx, by, bz), got {value!r}")
    for i, p in enumerate(value):
        flag(who, f"periodic[{i}]", p)


def choice(who, name, value, options):
    """One of `options`."""
    if value not in options:
        raise ValueError(f"{who}: {name} must be one of {', '.join(options)}, got {value!r}")


def solver(who, flow):
    """A peclet.flow solver (duck-typed on what the drivers call first)."""
    if not all(hasattr(flow, a) for a in ("set_dt", "spacing", "step")):
        raise TypeError(f"{who}: flow must be a peclet.flow.Solver, got {type(flow).__name__}")


def simulation(who, dem):
    """A peclet.dem.Simulation (duck-typed on what the drivers call first)."""
    if not all(hasattr(dem, a) for a in ("num_particles", "set_dt", "get_positions")):
        raise TypeError(f"{who}: dem must be a peclet.dem.Simulation, got {type(dem).__name__}")


def radius(who, value, n):
    """A scalar finite and > 0, or an (n,) array of such values (one per particle)."""
    if np.isscalar(value):
        positive(who, "radius", value, "length units")
        return
    a = np.asarray(value)
    if a.ndim != 1 or a.shape[0] != n or not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{who}: radius must be a scalar or an ({n},) array of radii "
                         f"(dem.num_particles = {n}), got shape {a.shape}")
    bad = ~(np.isfinite(a) & (a > 0))
    if bad.any():
        i = int(np.argmax(bad))
        raise ValueError(f"{who}: radius[{i}] must be finite and > 0 (length units), got {float(a[i])!r}")


def rows3(who, name, a, n=None):
    """An (n, 3) particle array; `n` None accepts any row count."""
    shape = getattr(a, "shape", None)
    if shape is None or len(shape) != 2 or shape[1] != 3 or (n is not None and shape[0] != n):
        want = "(N, 3)" if n is None else f"({n}, 3)"
        raise ValueError(f"{who}: {name} must be an {want} array, got shape {shape}")


def finite_rows(who, name, a, xp):
    """Every entry of a particle array finite (xp: numpy, or cupy on a device build)."""
    ok = xp.isfinite(a)
    if not bool(ok.all()):
        host = ok.get() if hasattr(ok, "get") else ok  # cupy -> numpy, on failure only
        i, j = (int(k) for k in np.argwhere(~host)[0])
        raise ValueError(f"{who}: {name}[{i}, {j}] must be finite (internal units), got {float(a[i, j])!r}")


# --- the constructors' argument sets -------------------------------------------------------------

def cfd_dem(a):
    """CfdDem.__init__'s arguments, as its locals() at entry."""
    who = "CfdDem"
    solver(who, a["flow"])
    simulation(who, a["dem"])
    positive(who, "fluid_dt", a["fluid_dt"], "time units")
    positive(who, "mu", a["mu"], "dynamic viscosity units, mass/(length*time)")
    positive(who, "rho", a["rho"], "density units, mass/length^3")
    radius(who, a["radius"], a["dem"].num_particles)
    choice(who, "drag", a["drag"], DRAG_LAWS)
    count(who, "dem_substeps", a["dem_substeps"], 1)
    fraction(who, "eps_min", a["eps_min"])
    nonneg(who, "smooth_length", a["smooth_length"], "length units; 0 disables the filter")
    periodic3(who, a["periodic"])
    for name in ("move_particles", "implicit_drag", "porous", "advection"):
        flag(who, name, a[name])
    finite3(who, "gravity", a["gravity"], "length/time^2 units")


def resolved_cfd_dem(a):
    """ResolvedCfdDem.__init__'s arguments, as its locals() at entry. periodic and force_method
    keep the driver's own checks (a mixed triple is a ValueError there)."""
    who = "ResolvedCfdDem"
    solver(who, a["flow"])
    simulation(who, a["dem"])
    positive(who, "radius", a["radius"], "length units")
    nonneg(who, "mu", a["mu"], "dynamic viscosity units, mass/(length*time)")
    positive(who, "rho", a["rho"], "density units, mass/length^3")
    positive(who, "fluid_dt", a["fluid_dt"], "time units")
    count(who, "dem_substeps", a["dem_substeps"], 1)
    finite3(who, "gravity", a["gravity"], "length/time^2 units")
    if a["rho_p"] is not None:
        positive(who, "rho_p", a["rho_p"], "density units, mass/length^3")
    for name in ("move_particles", "buoyancy", "apply_torque"):
        flag(who, name, a[name])
