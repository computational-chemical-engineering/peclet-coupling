"""Multi-rank CFD-DEM co-rebalancing onto the ALIGNED weighted ORB — CfdDem.rebalance().

A heap of slowly moving spheres sits in a driven periodic box. After two coupled steps on the
default decomposition, `CfdDem.rebalance(gamma=4)` builds one weight field (1 + 4 x particle count)
and moves BOTH codes onto the weighted ORB of it: flow through `rebalance_by_weights`, which picks
the partition's alignment 2^a for its pressure multigrid (the largest within the 1.05 imbalance
budget) and returns it, dem through `migrate_to_weights(w, align=2^a)`. Three more steps follow, in
which every moving step re-migrates dem with the same weights and alignment.

Checked:
  * the aligned path is exercised: at np > 1 the returned alignment is > 1 for this heap;
  * co-location holds: rebalance() and the first step assert that every particle dem owns lies in
    flow's block (a violation raises on every rank) -- reaching the end is the evidence;
  * the rebalance is physics-neutral: the mean particle and fluid x-velocities reproduce np = 1
    (where rebalance() is a no-op) to the pressure solve's reduction floor;
  * the assertion is not vacuous: migrating dem back onto the equal-cell ORB (a partition flow no
    longer owns) makes it raise, on every rank.

Run:  mpirun -np 1 python test_mpi_rebalance.py   (writes the reference; run it first)
      mpirun -np {2,4} python test_mpi_rebalance.py
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
    """Cell-centred lattice sites under a surface falling linearly in x and y (a heap), which moves
    every weighted-ORB split away from the equal-cell position."""
    c = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    zs = bed * N * (1.0 + tilt * (0.5 - X / N) + tilt * (0.5 - Y / N))
    occ = Z < zs
    return np.stack([X[occ], Y[occ], Z[occ]], axis=1)


def run(comm, N=32, r=0.3, before=2, after=3):
    m_p = (4.0 / 3.0) * np.pi * r ** 3
    (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(N, N, N)
    s = peclet.flow.Solver(lnx, lny, lnz)
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(0.1)
    s.set_body_force((0.05, 0.0, 0.0))
    s.init_mpi(N, N, N)  # BEFORE the geometry: flow raises otherwise
    s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 10.0)))

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
                 dem_substeps=5, move_particles=True)
    for _ in range(before):
        cpl.step()
    cpl.rebalance(gamma=4.0)   # 1 + 4 * count: a heap heavy enough to move every split
    align = cpl._align
    for _ in range(after):
        cpl.step()

    v = cpl._particles()[1]
    n = v.shape[0]
    gn = comm.allreduce(n, MPI.SUM)
    vp = comm.allreduce(float(v[:, 0].sum()) if n else 0.0, MPI.SUM) / gn
    uf = comm.allreduce(float(np.asarray(s.get_u()).sum()), MPI.SUM) / N ** 3

    # The negative control: dem back onto the equal-cell ORB (uniform weights), a partition flow
    # no longer owns after the heap rebalance -- particles now sit outside flow's block, and the
    # assertion must fire on every rank.
    fired = None
    if comm.Get_size() > 1:
        d.migrate_to_weights(np.ones(N ** 3), align=1)
        pos = cpl._particles()[0]
        try:
            cpl._assert_colocated(pos, "the negative control")
            fired = False
        except RuntimeError:
            fired = True
    return vp, uf, gn, align, fired


def test_mpi_rebalance():
    _require_mpi()
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    vp, uf, gn, align, fired = run(comm)
    ref_file = os.path.join(os.path.dirname(__file__), ".rebalance_ref.json")
    ok = bool(np.isfinite(vp) and np.isfinite(uf))
    if size == 1:
        if rank == 0:
            json.dump({"vp": vp, "uf": uf, "n": gn}, open(ref_file, "w"))
        tag = "reference"
    elif os.path.exists(ref_file):
        ref = json.load(open(ref_file))
        ev = abs(vp - ref["vp"]) / abs(ref["vp"])
        eu = abs(uf - ref["uf"]) / abs(ref["uf"])
        ok = ok and gn == ref["n"] and ev < 1e-6 and eu < 1e-6
        ok = ok and align > 1 and fired is True and comm.allreduce(int(fired), MPI.MIN) == 1
        tag = (f"align={align} vs np=1: particle vx rel-err={ev:.2e}, fluid u rel-err={eu:.2e}; "
               f"negative control {'raised' if fired else 'DID NOT RAISE'}")
    else:
        ok = False
        tag = "NO REFERENCE (run np=1 first)"
    if rank == 0:
        print(f"[np={size}] particles={gn} mean particle vx={vp:.10e} mean fluid u={uf:.10e}  {tag}")
        print(f"MPI REBALANCE (np={size}): {'PASS' if ok else 'FAIL'}")
    assert comm.allreduce(1 if ok else 0, MPI.MIN)


if __name__ == "__main__":
    test_mpi_rebalance()
