"""Gas-solid fluidized bed in a cylindrical vessel — a full CFD-DEM coupling example.

Demonstrates every requested ingredient of an unresolved point-particle fluidized bed:

  * a CYLINDRICAL vessel used for BOTH phases: an immersed no-slip wall for the gas (cut-cell IBM over
    the radial SDF) and a restitution+friction wall for the grains (dem SDF wall).
  * gas boundaries: INFLOW at the bottom face (superficial velocity U up), OUTFLOW at the top.
  * particle boundaries: a bouncy distributor at the bottom (restitution < 1) and a containment wall at
    the top so grains cannot leave -- while the gas passes straight through it (the dem wall is invisible
    to the flow, whose top face is an outflow).
  * SPHERICAL grains, cell size ~5x the particle diameter (the unresolved-CFD-DEM requirement).
  * the GIDASPOW drag correlation (Ergun in the dense regime eps<0.8, Wen & Yu in the dilute, switched
    at 0.8), with VOLUME-WEIGHTED porosity (trilinear solid-volume deposition -> void fraction).

Runs on device (Kokkos: OpenMP / CUDA / HIP) and is MPI-parallel: pass the same flow/dem decomposition
and CfdDem runs the distributed deposit-fold + gather + distributed DEM step (see run_mpi below).

    python fluidized_bed.py                 # single process (OpenMP or CUDA build)
    mpirun -np 4 python fluidized_bed.py    # 4 ranks (needs the PECLET_*_MPI modules)

Output: the bed height vs time -- it rises (the bed expands / fluidizes) once U exceeds the minimum
fluidization velocity.
"""
import numpy as np
import peclet.flow
import peclet.dem
try:
    from peclet.dem import build_wall_sdf
except ImportError:  # older builds don't re-export it from the package root
    from peclet.dem.particle_builder import build_wall_sdf
from peclet.coupling import CfdDem


# ----------------------------------------------------------------------------------------------------
# parameters (grid units: cell size h = 1; the bed sits in a box, cylinder axis along z)
# ----------------------------------------------------------------------------------------------------
class Params:
    NX = NY = 8           # lateral grid (cells)
    NZ = 18               # tall column (bed + freeboard)
    R = 2.5               # vessel radius (cells)
    H_wall = 12.0         # particle containment wall height (< NZ so the gas has a freeboard)
    dp = 0.2              # particle diameter -> cell size / dp = 5 (unresolved CFD-DEM)
    n_bed = 3.0           # initial loose-bed height (cells)
    solid_frac = 0.45     # initial packing solid fraction (settles into a packed bed)

    rho_g = 1.0           # gas density
    mu_g = 0.05           # gas viscosity
    rho_p = 40.0          # particle density (rho_p >> rho_g, like sand in air)
    g = 2.0e-3            # gravity
    U_in = 0.25           # superficial inflow velocity (> U_mf -> fluidizes)

    e_wall = 0.7          # particle-wall restitution (bouncy distributor != 1)
    mu_wall = 0.3         # particle-wall friction
    e_pp = 0.8            # particle-particle restitution
    mu_pp = 0.2           # particle-particle friction

    fluid_dt = 0.05
    dem_substeps = 20
    steps = 120


def cylinder_flow_sdf(P):
    """Flow IBM SDF on the inner grid (x-fastest): >0 in the fluid inside the vessel, <0 in the wall
    outside radius R. The bottom/top are OPEN (domain inflow/outflow faces), only the side confines."""
    cx, cy = P.NX / 2.0, P.NY / 2.0
    x = np.arange(P.NX) + 0.5
    y = np.arange(P.NY) + 0.5
    X, Y, Z = np.meshgrid(x, y, np.arange(P.NZ) + 0.5, indexing="ij")
    rad = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    return (P.R - rad).astype(np.float64)


def capped_cylinder_wall_sdf(P):
    """Particle wall f(points)->distance, >0 in the void where grains live: inside radius R, above the
    distributor (z>0), below the containment lid (z<H_wall)."""
    cx, cy = P.NX / 2.0, P.NY / 2.0

    def f(pts):
        rad = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        return np.minimum.reduce([P.R - rad, pts[:, 2] - 0.0, P.H_wall - pts[:, 2]])

    return f


def initial_packing(P, rp):
    """Loose sphere positions inside the vessel, z in (rp, n_bed]; settles into a packed bed."""
    cx, cy = P.NX / 2.0, P.NY / 2.0
    vol_bed = np.pi * P.R ** 2 * P.n_bed
    vp = (4.0 / 3.0) * np.pi * rp ** 3
    npart = int(P.solid_frac * vol_bed / vp)
    rng = np.random.default_rng(7)
    pos = np.empty((npart, 3), np.float32)
    n = 0
    while n < npart:                                   # rejection-sample inside the cylinder
        c = rng.uniform([cx - P.R, cy - P.R, rp], [cx + P.R, cy + P.R, P.n_bed], size=(npart, 3))
        ok = (c[:, 0] - cx) ** 2 + (c[:, 1] - cy) ** 2 < (P.R - rp) ** 2
        c = c[ok]
        take = min(len(c), npart - n)
        pos[n:n + take] = c[:take]
        n += take
    return pos, npart


def build(P, comm=None):
    rp = P.dp / 2.0
    m_p = P.rho_p * (4.0 / 3.0) * np.pi * rp ** 3
    mpi = comm is not None and comm.Get_size() > 1

    # --- flow: cylindrical no-slip vessel + gas inflow(bottom)/outflow(top) --------------------------
    if mpi:
        (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(P.NX, P.NY, P.NZ)
        s = peclet.flow.Solver(lnx, lny, lnz)
    else:
        s = peclet.flow.Solver(P.NX, P.NY, P.NZ)
    s.set_rho(P.rho_g); s.set_mu(P.mu_g); s.set_dt(P.fluid_dt)
    s.set_domain_bc(4, 2, 0.0, 0.0, P.U_in)   # -z face: inflow, gas velocity U up
    s.set_domain_bc(5, 3)                      # +z face: outflow
    for face in (0, 1, 2, 3):
        s.set_domain_bc(face, 1)               # x/y faces: no-slip (fluid never reaches them)
    gsdf = cylinder_flow_sdf(P)
    if mpi:
        s.init_mpi(P.NX, P.NY, P.NZ)
        lsdf = gsdf[ox:ox + lnx, oy:oy + lny, oz:oz + lnz]
        s.set_solid(np.asfortranarray(lsdf).flatten(order="F"), True)
    else:
        s.set_solid(gsdf.flatten(order="F"), True)

    # --- dem: spheres + capped-cylinder wall (restitution+friction) + gravity ------------------------
    pos, npart = initial_packing(P, rp)
    cap = int(2.2 * npart) + 256
    d = peclet.dem.Simulation(cap)
    d.initialize(shape_type=1, radius=rp)      # 1 = sphere, radius rp directly (SI-style: the DEM
    d.set_domain((0, 0, 0), (P.NX, P.NY, P.NZ))  # sizes its halo band from the actual grain radius)
    d.enable_periodicity(False, False, False)
    d.set_gravity(0.0, 0.0, -P.g)
    d.set_material_params(P.e_pp, 0.0, P.mu_pp)
    d.set_dt(P.fluid_dt / P.dem_substeps)
    wall = build_wall_sdf(capped_cylinder_wall_sdf(P),
                          ((0, 0, 0), (P.NX, P.NY, P.NZ)), resolution=64)
    wall.add_to(d, restitution=P.e_wall, friction=P.mu_wall)

    posw = np.concatenate([pos, np.full((npart, 1), 1.0 / m_p, np.float32)], axis=1)
    if mpi:
        keep = ((np.floor(pos[:, 0]) >= ox) & (np.floor(pos[:, 0]) < ox + lnx) &
                (np.floor(pos[:, 1]) >= oy) & (np.floor(pos[:, 1]) < oy + lny) &
                (np.floor(pos[:, 2]) >= oz) & (np.floor(pos[:, 2]) < oz + lnz))
        posw = posw[keep]
        d.set_positions(posw)
        d.set_velocities(np.zeros((posw.shape[0], 3), np.float32))
        d.init_mpi((0.0, 0.0, 0.0), (float(P.NX), float(P.NY), float(P.NZ)),
                   (P.NX, P.NY, P.NZ), (False, False, False))
        d.enable_mpi_step(2.0 * rp, rebalance_every=0)
    else:
        d.set_positions(posw)
        d.set_velocities(np.zeros((npart, 3), np.float32))

    cpl = CfdDem(s, d, fluid_dt=P.fluid_dt, mu=P.mu_g, rho=P.rho_g, radius=rp, drag="gidaspow",
                 dem_substeps=P.dem_substeps, eps_min=0.3, periodic=(False, False, False),
                 move_particles=True)
    return s, d, cpl, npart


def _np(a):
    """A NumPy view of a host or device (CuPy) array."""
    return a.get() if hasattr(a, "get") and type(a).__module__.startswith("cupy") else np.asarray(a)


def bed_height(cpl, comm=None):
    """95th-percentile particle height (the top of the bed) -- rises when the bed fluidizes. Uses the
    driver's device-safe position getter (dem's host copy getter breaks after a resizing step on CUDA)."""
    try:
        z = _np(cpl._particles()[0])[:, 2]
        h = float(np.percentile(z, 95)) if z.size else 0.0
    except Exception:
        h = 0.0
    if comm is not None:
        h = comm.allreduce(h, op=__import__("mpi4py").MPI.MAX)
    return h


def run(P=Params(), comm=None):
    rank = comm.Get_rank() if comm is not None else 0
    s, d, cpl, npart = build(P, comm)
    if rank == 0:
        print(f"fluidized bed: {npart} grains, dp={P.dp} (cell/dp={1.0/P.dp:.0f}), R={P.R}, "
              f"U={P.U_in}, Gidaspow drag", flush=True)
    h0 = bed_height(cpl, comm)
    for i in range(P.steps):
        cpl.step()
        if i % 40 == 0 or i == P.steps - 1:
            h = bed_height(cpl, comm)
            eps = cpl.last_eps
            emin = float(_np(eps).min()) if eps is not None else 1.0
            if rank == 0:
                print(f"  step {i:4d}: bed height={h:6.3f}  (h/h0={h/max(h0,1e-9):.2f})  "
                      f"min voidage={emin:.3f}", flush=True)
    hf = bed_height(cpl, comm)
    if rank == 0:
        print(f"DONE: bed expanded h0={h0:.3f} -> hf={hf:.3f} (ratio {hf/max(h0,1e-9):.2f}); "
              f"{'FLUIDIZED' if hf > 1.05 * h0 else 'fixed bed (increase U_in)'}", flush=True)
    return h0, hf


def sweep(comm=None):
    """Fluidization curve: raise the gas velocity and watch the bed height rise once U > U_mf."""
    rank = comm.Get_rank() if comm is not None else 0
    if rank == 0:
        print("=== fluidization sweep (bed height vs superficial gas velocity) ===", flush=True)
    for U in (0.0, 0.08, 0.16, 0.30):
        P = Params()
        P.U_in = U
        h0, hf = run(P, comm)
        if rank == 0:
            print(f"  U={U:.2f}: bed height {hf:.2f}  ({'fluidized' if hf > 1.05 * h0 else 'fixed'})\n",
                  flush=True)


if __name__ == "__main__":
    import sys
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD if MPI.COMM_WORLD.Get_size() > 1 else None
    except Exception:
        comm = None
    if "--sweep" in sys.argv:
        sweep(comm)
    else:
        run(Params(), comm)
