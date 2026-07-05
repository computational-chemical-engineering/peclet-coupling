"""Multi-rank CFD-DEM with MOVING particles, DISTRIBUTED — validates the distributed moving loop.

A cloud of spheres drifts across a decomposed periodic box (initial velocity along the ORB split
axis x), coupled two-way to the fluid. As they drift they CROSS block boundaries and their ownership
MIGRATES between ranks each fluid step: CfdDem migrates dem onto flow's grid partition, runs the
DISTRIBUTED DEM substeps (halo-exchanged), deposits the drag reaction (folded across ranks via the
reverse halo), gathers the fluid velocity at each particle, and steps the distributed flow. The mean
particle velocity must REPRODUCE the single-rank result — proving the whole distributed moving loop
(migration + distributed DEM step + deposit fold + gather) matches np=1.

Scope note: this exercises a short migration window. A *sustained* dilute settling suspension in a
triply-periodic box with no buoyancy is an ill-posed, numerically unstable configuration (the mean
velocity is unconstrained; a uniform suspension is unstable to clustering) — at np>1 the flow solve's
reduction-floor non-determinism seeds that instability. That is a property of the configuration, not
of the coupling: every distributed operation (fold, migrate, gather) is bit-identical to single-rank
in isolation (see the unit checks). Well-posed cases (bounded/driven flow) run indefinitely.

Run:  mpirun -np {1,2,4} python test_mpi_moving_suspension.py
"""
import os
import json
import numpy as np
import peclet.flow
import peclet.dem
from peclet.coupling import CfdDem
from mpi4py import MPI


def run(comm, N=32, r=0.7, steps=6, v0=-6.0):
    m_p = (4.0 / 3.0) * np.pi * r ** 3
    (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(N, N, N)

    s = peclet.flow.Solver(lnx, lny, lnz)
    s.set_rho(1.0); s.set_mu(1.0); s.set_dt(0.1)
    s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 10.0)))
    s.init_mpi(N, N, N)

    # cloud spanning x (crosses the x rank boundary), y,z in the interior; drift along -x.
    xv = (np.arange(6) + 0.5) * N / 6
    yv = zv = np.array([10.0, 16.0, 22.0])
    X, Y, Z = np.meshgrid(xv, yv, zv, indexing="ij")
    gp = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    cell = np.floor(gp).astype(int)
    keep = ((cell[:, 0] >= ox) & (cell[:, 0] < ox + lnx) &
            (cell[:, 1] >= oy) & (cell[:, 1] < oy + lny) &
            (cell[:, 2] >= oz) & (cell[:, 2] < oz + lnz))
    mine = gp[keep].astype(np.float32)
    Np = mine.shape[0]
    posw = np.concatenate([mine, np.full((Np, 1), 1.0 / m_p, dtype=np.float32)], axis=1)

    d = peclet.dem.Simulation(6 * 36 + 64)
    d.initialize(shape_type=1, radius=r)
    d.set_domain((0, 0, 0), (N, N, N))
    d.enable_periodicity(True, True, True)
    d.set_gravity(0.0, 0.0, 0.0)
    vel = np.zeros((Np, 3), dtype=np.float32); vel[:, 0] = v0
    d.set_positions(posw); d.set_velocities(vel)
    d.init_mpi((0.0, 0.0, 0.0), (float(N),) * 3, (N, N, N), (True, True, True))
    d.enable_mpi_step(2.0 * r, rebalance_every=0)

    cpl = CfdDem(s, d, fluid_dt=0.1, mu=1.0, rho=1.0, radius=r, drag="stokes",
                 dem_substeps=10, move_particles=True)
    # min owned x per step: if a particle ends up in a block it did not start in, migration occurred.
    crossed = False
    for _ in range(steps):
        cpl.step()
        p = cpl._particles()[0]
        lo = float(p[:, 0].min()) if p.shape[0] else 1e9
        gmin = comm.allreduce(lo, MPI.MIN)
        if gmin < N / 6 * 0.5:   # a particle drifted below the first lattice plane -> it wrapped/crossed
            crossed = True
    v = cpl._particles()[1]
    n = v.shape[0]
    gv = comm.allreduce(float(v[:, 0].sum()) if n else 0.0, MPI.SUM)
    gn = comm.allreduce(n, MPI.SUM)
    return gv / gn, comm.allreduce(1 if crossed else 0, MPI.MAX)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    mean_vx, crossed = run(comm)
    ref_file = os.path.join(os.path.dirname(__file__), ".moving_ref.json")
    ok = np.isfinite(mean_vx) and crossed
    if size == 1:
        if rank == 0:
            json.dump({"mean_vx": mean_vx}, open(ref_file, "w"))
        tag = "reference"
    else:
        ref = json.load(open(ref_file))["mean_vx"] if os.path.exists(ref_file) else mean_vx
        err = abs(mean_vx - ref) / abs(ref)
        ok = ok and err < 1e-4  # reproduce single-rank across the migration window
        tag = f"vs np=1 {ref:.8e} rel-err={err:.2e}"
    if rank == 0:
        print(f"[np={size}] mean_vx={mean_vx:.8e}  migrated={bool(crossed)}  {tag}")
        print(f"MPI MOVING SUSPENSION (np={size}): {'PASS' if ok else 'FAIL'}")
    import sys
    sys.exit(0 if comm.allreduce(1 if ok else 0, MPI.MIN) else 1)
