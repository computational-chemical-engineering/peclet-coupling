"""Multi-rank CFD-DEM, MOVING and POLYDISPERSE: each particle's radius follows it through every
migration (the construction co-rebalance and the per-step migrate_to_weights).

The drag depends on the particle radius, so a radius array that stays in the rank's pre-migration
order hands every migrated particle some other particle's radius (and, when the owned count
changes, an array of the wrong length). CfdDem keeps no such array under MPI: dem carries each
particle's size as its scale, and the driver re-derives the owned radii from dem after every
migration. A 48^3 box (the construction co-rebalance moves every split) and a drifting cloud of
two radii, 0.5 and 0.8, crossing rank boundaries. Checks:

  * every particle's coupling radius matches its own mass (dem migrates the inverse mass with the
    particle; rho_p = 1, so r = (3 m / 4 pi)^(1/3)) after the construction and after every step;
  * the mean particle x-velocity reproduces np = 1 (where nothing migrates).

Run:  mpirun -np 1 python test_mpi_polydisperse_moving.py   (writes the reference; run it first)
      mpirun -np {2,4} python test_mpi_polydisperse_moving.py
"""
import os
import json
import numpy as np
import peclet.flow
import peclet.dem
from peclet.coupling import CfdDem
try:
    from mpi4py import MPI
except ImportError:  # a host without mpi4py: the pytest entry skips, the script entry fails loudly
    MPI = None


def _require_mpi():
    """The distributed tests need flow's MPI build + mpi4py; under pytest they SKIP otherwise
    (never silently green), as a script they still run: mpirun -np N python <this file>."""
    if MPI is None or not getattr(peclet.flow, "has_mpi", False):
        import pytest
        pytest.skip("needs mpi4py + a PECLET_FLOW_MPI build of peclet.flow (run: mpirun -np N python ...)")


def radius_mismatch(cpl, d):
    """Largest |coupling radius - radius implied by the particle's own mass| over owned particles."""
    m = np.asarray(d.get_masses(), dtype=np.float64)
    if m.size == 0:
        return 0.0
    r_mass = (3.0 * m / (4.0 * np.pi)) ** (1.0 / 3.0)
    rad = cpl._rad.get() if cpl.device else np.asarray(cpl._rad)
    if rad.shape[0] != m.size:
        return np.inf
    return float(np.abs(rad.astype(np.float64) - r_mass).max())


def run(comm, N=48, steps=6, v0=-6.0, radii=(0.5, 0.8)):
    (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(N, N, N)

    s = peclet.flow.Solver(lnx, lny, lnz)
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(0.1)
    s.init_mpi(N, N, N)  # BEFORE the geometry: flow raises otherwise
    s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 10.0)))

    # cloud spanning x (crosses the x rank boundaries), drifting along -x; radii alternate.
    xv = (np.arange(6) + 0.5) * N / 6
    yv = zv = np.array([15.0, 24.0, 33.0])
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing="ij")
    gp = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    gr = np.where(np.arange(gp.shape[0]) % 2 == 0, radii[0], radii[1])
    cell = np.floor(gp).astype(int)
    keep = ((cell[:, 0] >= ox) & (cell[:, 0] < ox + lnx) &
            (cell[:, 1] >= oy) & (cell[:, 1] < oy + lny) &
            (cell[:, 2] >= oz) & (cell[:, 2] < oz + lnz))
    mine = gp[keep].astype(np.float32)
    r = gr[keep]
    Np = mine.shape[0]
    m_p = (4.0 / 3.0) * np.pi * r ** 3
    posw = np.concatenate([mine, (1.0 / m_p).astype(np.float32)[:, None]], axis=1)

    d = peclet.dem.Simulation(gp.shape[0] + 64)
    d.initialize_shape('sphere', radius=1.0)   # world radius = scale x 1
    d.set_domain(extent=(N, N, N), periodic=(True, True, True))
    d.set_gravity((0.0, 0.0, 0.0))
    vel = np.zeros((Np, 3), dtype=np.float32); vel[:, 0] = v0
    d.set_positions(posw); d.set_velocities(vel)
    d.set_scales(r.astype(np.float32))          # after set_positions, which sizes the set
    d.init_mpi((0.0, 0.0, 0.0), (float(N),) * 3, (N, N, N), (True, True, True))
    d.enable_mpi_step(2.0 * max(radii), rebalance_every=0)

    cpl = CfdDem(s, d, fluid_dt=0.1, mu=1.0, rho=1.0, radius=r, drag="stokes",
                 dem_substeps=10, move_particles=True)
    err = radius_mismatch(cpl, d)
    for _ in range(steps):
        cpl.step()
        err = max(err, radius_mismatch(cpl, d))
    err = comm.allreduce(err, MPI.MAX)
    v = cpl._particles()[1]
    n = v.shape[0]
    gv = comm.allreduce(float(v[:, 0].sum()) if n else 0.0, MPI.SUM)
    gn = comm.allreduce(n, MPI.SUM)
    return gv / gn, gn, err


def test_mpi_polydisperse_moving():
    _require_mpi()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    mean_vx, gn, rerr = run(comm)
    ref_file = os.path.join(os.path.dirname(__file__), ".polydisperse_ref.json")
    ok = bool(np.isfinite(mean_vx)) and rerr < 1e-5
    if size == 1:
        if rank == 0:
            json.dump({"mean_vx": mean_vx, "n": gn}, open(ref_file, "w"))
        tag = "reference"
    elif os.path.exists(ref_file):
        ref = json.load(open(ref_file))
        e = abs(mean_vx - ref["mean_vx"]) / abs(ref["mean_vx"])
        ok = ok and gn == ref["n"] and e < 1e-4
        tag = f"vs np=1 {ref['mean_vx']:.8e} rel-err={e:.2e}"
    else:
        ok = False
        tag = "NO REFERENCE (run np=1 first)"
    if rank == 0:
        print(f"[np={size}] mean_vx={mean_vx:.8e} max|r - r(m)|={rerr:.2e}  {tag}")
        print(f"MPI POLYDISPERSE MOVING (np={size}): {'PASS' if ok else 'FAIL'}")
    assert comm.allreduce(1 if ok else 0, MPI.MIN)


if __name__ == "__main__":
    test_mpi_polydisperse_moving()
