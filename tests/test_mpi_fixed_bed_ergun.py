"""Multi-rank CFD-DEM: fixed-bed pressure drop vs Ergun, DISTRIBUTED.

The single-rank fixed-bed Ergun benchmark (test_fixed_bed_ergun.py) run across MPI ranks: the flow
solver is decomposed (Solver.init_mpi), each rank holds the lattice particles in its ORB block, and
the coupling deposits/gathers on the LOCAL block with a block-origin-shifted grid map — cross-rank +
periodic ghost deposits (void fraction + drag reaction) fold onto their owner via the reverse
(add-reduce) halo (flow.exchange_field_add). The measured superficial velocity U (reduced over ranks)
must land on the same Ergun curve as single-rank, proving the distributed deposition + fold + solve
reproduce the coupled physics.

Run:  mpirun -np {1,2,4} python test_mpi_fixed_bed_ergun.py
"""
import numpy as np
import peclet.flow
import peclet.dem
from peclet.coupling import CfdDem
from mpi4py import MPI


def ergun(U, eps, mu, rho, d):
    om = 1.0 - eps
    return (150.0 * mu * om * om / (eps ** 3 * d * d) * U
            + 1.75 * rho * om / (eps ** 3 * d) * U * U)


def run_bed(f_drive, comm, eps_target=0.6, N=16, mu=1.0, rho=1.0, dt=0.5, steps=120):
    rank, size = comm.Get_rank(), comm.Get_size()
    Vp = 1.0 - eps_target  # Vcell = 1 (h=1)
    r = (Vp * 3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)

    # this rank's ORB block of the global N^3 grid
    (ox, oy, oz), (lnx, lny, lnz) = peclet.flow.mpi_block(N, N, N)

    # global lattice (one particle per cell centre); keep those whose cell is in this block.
    xs = np.arange(N) + 0.5
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing="ij")
    gpos = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(np.float32)
    cell = np.floor(gpos).astype(int)
    keep = ((cell[:, 0] >= ox) & (cell[:, 0] < ox + lnx) &
            (cell[:, 1] >= oy) & (cell[:, 1] < oy + lny) &
            (cell[:, 2] >= oz) & (cell[:, 2] < oz + lnz))
    mine = gpos[keep]
    Np = mine.shape[0]
    posw = np.concatenate([mine, np.zeros((Np, 1), dtype=np.float32)], axis=1)  # invMass 0 => fixed

    s = peclet.flow.Solver(lnx, lny, lnz)
    s.set_rho(rho); s.set_mu(mu); s.set_dt(dt)
    s.set_body_force(0.0, 0.0, f_drive)
    s.init_mpi(N, N, N)
    s.set_pressure_geometry(np.asfortranarray(np.full((lnx, lny, lnz), 10.0)))

    d = peclet.dem.Simulation(max(Np, 1))
    d.initialize(shape_type=1, radius=r)
    d.set_domain((0, 0, 0), (N, N, N))
    d.enable_periodicity(True, True, True)
    d.set_positions(posw)
    d.set_velocities(np.zeros((Np, 3), dtype=np.float32))

    cpl = CfdDem(s, d, fluid_dt=dt, mu=mu, rho=rho, radius=r, drag="ergun", eps_min=0.05,
                 move_particles=False)
    for _ in range(steps):
        cpl.step()

    # global superficial velocity U + mean void fraction, reduced over ranks.
    lw = float(np.asarray(s.get_w()).sum())
    gw = comm.allreduce(lw, MPI.SUM)
    U = gw / (N * N * N)
    ep = cpl.last_eps
    inner = np.asarray(ep[cpl.g:cpl.g + lnx, cpl.g:cpl.g + lny, cpl.g:cpl.g + lnz])
    leps = float(inner.sum())
    geps = comm.allreduce(leps, MPI.SUM)
    eps_mean = geps / (N * N * N)
    return U, eps_mean, ergun(U, eps_mean, mu, rho, 2 * r), r


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    if rank == 0:
        print(f"[np={size}]  f_drive     U         eps    Ergun(U,eps)  rel-err")
    ok = True
    for f_drive in (0.2, 20.0, 1000.0):
        U, eps, dP, r = run_bed(f_drive, comm)
        err = abs(dP - f_drive) / f_drive
        Re = 2 * r * abs(U) / 1.0
        if rank == 0:
            print(f"          {f_drive:8.2f}  {U:.4e}  {eps:.3f}  {dP:.4e}   {err*100:5.1f}%  "
                  f"(Re_p~{Re:.2f})")
        ok = ok and err < 0.10
    ok = comm.allreduce(1 if ok else 0, MPI.MIN)
    if rank == 0:
        print(f"MPI FIXED-BED ERGUN (np={size}): {'PASS' if ok else 'FAIL'}")
    import sys
    sys.exit(0 if ok else 1)
