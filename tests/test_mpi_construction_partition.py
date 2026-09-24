"""Multi-rank CFD-DEM owns ONE partition from construction on -- no rebalance() by hand.

On a grid that is not a power of two, flow's `init_mpi` partition snaps its splits to powers of two
and dem's equal-cell ORB does not (48^3 at np = 4: flow 32|16, dem 24|24), so before any rebalance
the two codes owned different blocks and particles were deposited into cells their rank does not
own. A moving multi-rank `CfdDem` now co-rebalances both codes at construction onto the aligned
weighted ORB of the combined cost field 1 + gamma x (particles in the cell), gamma the default.

A 48^3 heap, then a few moving steps, never calling rebalance(). Checks:

  * co-location holds: construction and the first step assert it on every rank (a violation raises);
  * the construction really re-partitioned: at np > 1 flow's block is no longer its init_mpi block
    on at least one rank, and the alignment flow chose is recorded;
  * the run is physics-neutral: the mean particle and fluid x-velocities reproduce np = 1 (where
    the construction co-rebalance is skipped) to the pressure solve's reduction floor.

Run:  mpirun -np 1 python test_mpi_construction_partition.py   (writes the reference; run it first)
      mpirun -np {2,4} python test_mpi_construction_partition.py
"""
import json
import os

import numpy as np
import peclet.dem
import peclet.flow
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


def heap(N, bed=0.35, tilt=0.8):
    """Cell-centred lattice sites under a surface falling linearly in x and y (a heap)."""
    c = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    zs = bed * N * (1.0 + tilt * (0.5 - X / N) + tilt * (0.5 - Y / N))
    occ = Z < zs
    return np.stack([X[occ], Y[occ], Z[occ]], axis=1)


def run(comm, N=48, r=0.3, steps=3):
    m_p = (4.0 / 3.0) * np.pi * r ** 3
    (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(N, N, N)
    s = peclet.flow.Solver(lnx, lny, lnz)
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(0.1)
    s.set_body_force((0.05, 0.0, 0.0))
    s.init_mpi(N, N, N)  # BEFORE the geometry: flow raises otherwise
    s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 10.0)))

    # the particles are handed to the rank whose FLOW init_mpi block holds them; dem's own init_mpi
    # block differs on this grid, and the construction co-rebalance must sort that out.
    gp = heap(N)
    cell = np.floor(gp).astype(int)
    keep = ((cell[:, 0] >= ox) & (cell[:, 0] < ox + lnx) &
            (cell[:, 1] >= oy) & (cell[:, 1] < oy + lny) &
            (cell[:, 2] >= oz) & (cell[:, 2] < oz + lnz))
    mine = gp[keep].astype(np.float32)
    Np = mine.shape[0]
    posw = np.concatenate([mine, np.full((Np, 1), 1.0 / m_p, dtype=np.float32)], axis=1)
    d = peclet.dem.Simulation(gp.shape[0] + 64)
    d.initialize_shape('sphere', radius=r)
    d.set_domain(extent=(N, N, N), periodic=(True, True, True))
    d.set_gravity((0.0, 0.0, 0.0))
    vel = np.zeros((Np, 3), dtype=np.float32); vel[:, 0] = 0.5
    d.set_positions(posw); d.set_velocities(vel)
    d.init_mpi((0.0, 0.0, 0.0), (float(N),) * 3, (N, N, N), (True, True, True))
    d.enable_mpi_step(2.0 * r, rebalance_every=0)

    cpl = CfdDem(s, d, fluid_dt=0.1, mu=1.0, rho=1.0, radius=r, drag="stokes",
                 dem_substeps=5, move_particles=True)   # co-rebalances here (np > 1)
    moved = tuple(cpl._blo) != (ox, oy, oz) or tuple(s.cells) != (lnx, lny, lnz)
    moved = comm.allreduce(int(moved), MPI.MAX)
    for _ in range(steps):
        cpl.step()

    v = cpl._particles()[1]
    n = v.shape[0]
    gn = comm.allreduce(n, MPI.SUM)
    vp = comm.allreduce(float(v[:, 0].sum()) if n else 0.0, MPI.SUM) / gn
    uf = comm.allreduce(float(np.asarray(s.get_u()).sum()), MPI.SUM) / N ** 3
    return vp, uf, gn, cpl._align, bool(moved)


def test_mpi_construction_partition():
    _require_mpi()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    vp, uf, gn, align, moved = run(comm)
    ref_file = os.path.join(os.path.dirname(__file__), ".construction_partition_ref.json")
    ok = bool(np.isfinite(vp) and np.isfinite(uf))
    if size == 1:
        if rank == 0:
            json.dump({"vp": vp, "uf": uf, "n": gn}, open(ref_file, "w"))
        tag = "reference"
    elif os.path.exists(ref_file):
        ref = json.load(open(ref_file))
        ev = abs(vp - ref["vp"]) / abs(ref["vp"])
        eu = abs(uf - ref["uf"]) / abs(ref["uf"])
        ok = ok and gn == ref["n"] and ev < 1e-6 and eu < 1e-6 and moved
        tag = (f"align={align} re-partitioned={moved} vs np=1: particle vx rel-err={ev:.2e}, "
               f"fluid u rel-err={eu:.2e}")
    else:
        ok = False
        tag = "NO REFERENCE (run np=1 first)"
    if rank == 0:
        print(f"[np={size}] particles={gn} mean particle vx={vp:.10e} mean fluid u={uf:.10e}  {tag}")
        print(f"MPI CONSTRUCTION PARTITION (np={size}): {'PASS' if ok else 'FAIL'}")
    assert comm.allreduce(1 if ok else 0, MPI.MIN)


if __name__ == "__main__":
    test_mpi_construction_partition()
