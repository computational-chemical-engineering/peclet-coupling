/// @file
/// @brief coupling — CFD-DEM exchange kernels: void-fraction deposition + fused drag/feedback.
///
/// Built on core's trilinear particle<->grid primitive (interp/particle_grid.hpp) and the drag laws
/// (drag.hpp). Two device operations per fluid step:
///   1. depositSolidVolume + voidFraction: scatter each particle's volume to the grid -> eps.
///   2. computeDragFeedback (fused): gather the fluid velocity + eps at each particle, evaluate the
///      drag law, write the drag FORCE to the particle array (-> dem extForce), and scatter the
///      equal-opposite reaction as a force DENSITY (-F/Vcell) onto the fluid momentum-source fields
///      (-> flow cellForce). Momentum is conserved: sum_particles F == -sum_cells feedback*Vcell.
/// Atomics make the deposits order-dependent (tolerance-, not bit-exact).
#ifndef PECLET_COUPLING_COUPLING_KERNELS_HPP
#define PECLET_COUPLING_COUPLING_KERNELS_HPP

#include <Kokkos_Core.hpp>

#include "drag.hpp"
#include "peclet/core/interp/particle_grid.hpp"

namespace peclet::coupling {

using peclet::core::interp::GridMap;

// A stencil corner is a FLUID cell iff the (cell-centred) SDF there is >= 0 (SDF < 0 inside the
// solid, per docs/CONVENTIONS.md). With no geometry set, flow's sdf field is all-zero -> every corner
// reads as fluid -> the wall-aware paths below reduce EXACTLY to plain trilinear (the periodic no-wall
// case is byte-identical).
template <class MaskV>
KOKKOS_INLINE_FUNCTION bool cornerIsFluid(const MaskV& sdf, long o) {
  return (double)sdf(o) >= 0.0;
}

// Wall-aware trilinear gather: interpolate a cell-centred field at the particle using ONLY the fluid
// corners, reweighted to a partition of unity (a solid corner carries no data — its velocity is the
// no-slip 0 and its eps is meaningless — so including it biases the interpolant). Reduces to plain
// trilinear when all 8 corners are fluid. `fallback` is returned in the degenerate all-solid case
// (particle centre inside the solid — should not happen for a physical bed).
template <class FieldV, class MaskV>
KOKKOS_INLINE_FUNCTION double gatherAtMasked(const FieldV& f, const MaskV& sdf, long b, long sx,
                                             long sy, long sz, double wx, double wy, double wz,
                                             double fallback) {
  const double w[2] = {1.0 - wx, wx}, wj[2] = {1.0 - wy, wy}, wk[2] = {1.0 - wz, wz};
  double acc = 0.0, tot = 0.0;
  for (int dk = 0; dk < 2; ++dk)
    for (int dj = 0; dj < 2; ++dj)
      for (int di = 0; di < 2; ++di) {
        const long o = b + di * sx + dj * sy + dk * sz;
        if (cornerIsFluid(sdf, o)) {
          const double wgt = w[di] * wj[dj] * wk[dk];
          acc += wgt * (double)f(o);
          tot += wgt;
        }
      }
  return (tot > 0.0) ? acc / tot : fallback;
}

// Wall-aware scatter of a conserved quantity q (solid volume, or a momentum-source density) onto the
// 8 stencil corners: distribute ONLY over the fluid corners, reweighting them to a partition of unity
// so a corner that falls inside the solid hands its share to the fluid corners instead of leaking q
// into a cell the fluid solver never sees. sum of deposits == q exactly (mass/momentum conserved).
// Reduces to plain trilinear when all 8 corners are fluid; degenerate all-solid dumps q at the base
// corner (conserves q). The eps clamp downstream keeps the concentrated hold-up from driving eps < 0.
template <class FieldV, class MaskV>
KOKKOS_INLINE_FUNCTION void scatterAtMasked(const FieldV& f, const MaskV& sdf, long b, long sx,
                                            long sy, long sz, double wx, double wy, double wz,
                                            double q) {
  using T = typename FieldV::value_type;
  const double w[2] = {1.0 - wx, wx}, wj[2] = {1.0 - wy, wy}, wk[2] = {1.0 - wz, wz};
  double tot = 0.0;
  for (int dk = 0; dk < 2; ++dk)
    for (int dj = 0; dj < 2; ++dj)
      for (int di = 0; di < 2; ++di) {
        const long o = b + di * sx + dj * sy + dk * sz;
        if (cornerIsFluid(sdf, o))
          tot += w[di] * wj[dj] * wk[dk];
      }
  if (tot <= 0.0) {  // particle centre buried in solid: conserve q by dumping it at the base corner
    Kokkos::atomic_add(&f(b), (T)q);
    return;
  }
  const double inv = 1.0 / tot;
  for (int dk = 0; dk < 2; ++dk)
    for (int dj = 0; dj < 2; ++dj)
      for (int di = 0; di < 2; ++di) {
        const long o = b + di * sx + dj * sy + dk * sz;
        if (cornerIsFluid(sdf, o))
          Kokkos::atomic_add(&f(o), (T)(q * w[di] * wj[dj] * wk[dk] * inv));
      }
}

// Scatter each particle's volume (4/3 pi r^3) onto `solidvol` (pre-zeroed). Trilinear -> conserves
// the total solid volume.
template <class PosV, class RadV, class FieldV>
void depositSolidVolume(int np, PosV pos, RadV rad, FieldV solidvol, FieldV sdf, GridMap m) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int nx = m.ex - 2 * m.g, ny = m.ey - 2 * m.g, nz = m.ez - 2 * m.g;
  Kokkos::parallel_for(
      "peclet::coupling::deposit_vol", Kokkos::RangePolicy<Exec>(0, np), KOKKOS_LAMBDA(int p) {
        int i0, j0, k0;
        double wx, wy, wz;
        peclet::core::interp::detail::axisStencil((double)pos(p, 0), m.ox, m.idx, nx, i0, wx);
        peclet::core::interp::detail::axisStencil((double)pos(p, 1), m.oy, m.idy, ny, j0, wy);
        peclet::core::interp::detail::axisStencil((double)pos(p, 2), m.oz, m.idz, nz, k0, wz);
        const long sx = 1, sy = m.ex, sz = (long)m.ex * m.ey;
        const long b = (long)(i0 + m.g) + (long)(j0 + m.g) * sy + (long)(k0 + m.g) * sz;
        const double r = (double)rad(p);
        // Wall-aware: the solid hold-up goes only to fluid cells (partition of unity), so no particle
        // volume leaks into the solid where the fluid solver would never account for it.
        scatterAtMasked(solidvol, sdf, b, sx, sy, sz, wx, wy, wz, (4.0 / 3.0) * M_PI * r * r * r);
      });
}

// eps = clamp(1 - solidvol/Vcell, epsMin, 1) over the whole (padded) field.
template <class FieldV>
void voidFraction(FieldV solidvol, FieldV eps, double invVcell, double epsMin) {
  using Exec = Kokkos::DefaultExecutionSpace;
  Kokkos::parallel_for(
      "peclet::coupling::void_fraction", Kokkos::RangePolicy<Exec>(0, (long)eps.extent(0)),
      KOKKOS_LAMBDA(long i) {
        double e = 1.0 - (double)solidvol(i) * invVcell;
        if (e < epsMin)
          e = epsMin;
        if (e > 1.0)
          e = 1.0;
        eps(i) = (typename FieldV::value_type)e;
      });
}

// Fused drag + feedback. Gathers (uf,vf,wf,eps) at each particle, evaluates the drag law, writes the
// drag force to fdrag(p,:) (dem external force), and scatters the reaction -F*invVcell onto the
// (pre-zeroed) grid force-density fields fx,fy,fz. rho/mu physical; dragKind per drag.hpp.
template <class PosV, class VelV, class RadV, class FieldV, class OutV>
void computeDragFeedback(int np, PosV pos, VelV vel, RadV rad, FieldV uf, FieldV vf, FieldV wf,
                         FieldV eps, FieldV sdf, OutV fdrag, FieldV fx, FieldV fy, FieldV fz,
                         GridMap m, double mu, double rhof, double invVcell, int dragKind,
                         bool modelB) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int nx = m.ex - 2 * m.g, ny = m.ey - 2 * m.g, nz = m.ez - 2 * m.g;
  Kokkos::parallel_for(
      "peclet::coupling::drag_feedback", Kokkos::RangePolicy<Exec>(0, np), KOKKOS_LAMBDA(int p) {
        int i0, j0, k0;
        double wx, wy, wz;
        peclet::core::interp::detail::axisStencil((double)pos(p, 0), m.ox, m.idx, nx, i0, wx);
        peclet::core::interp::detail::axisStencil((double)pos(p, 1), m.oy, m.idy, ny, j0, wy);
        peclet::core::interp::detail::axisStencil((double)pos(p, 2), m.oz, m.idz, nz, k0, wz);
        const long sx = 1, sy = m.ex, sz = (long)m.ex * m.ey;
        const long b = (long)(i0 + m.g) + (long)(j0 + m.g) * sy + (long)(k0 + m.g) * sz;
        // Wall-aware gather: interpolate from fluid corners only (a solid corner's velocity is the
        // no-slip 0 and its eps is meaningless).
        const double uF = gatherAtMasked(uf, sdf, b, sx, sy, sz, wx, wy, wz, 0.0);
        const double vF = gatherAtMasked(vf, sdf, b, sx, sy, sz, wx, wy, wz, 0.0);
        const double wF = gatherAtMasked(wf, sdf, b, sx, sy, sz, wx, wy, wz, 0.0);
        const double eP = gatherAtMasked(eps, sdf, b, sx, sy, sz, wx, wy, wz, 1.0);
        const double vrx = uF - (double)vel(p, 0), vry = vF - (double)vel(p, 1),
                     vrz = wF - (double)vel(p, 2);
        double Fx, Fy, Fz;
        dragForce(dragKind, vrx, vry, vrz, (double)rad(p), mu, rhof, eP, Fx, Fy, Fz);
        if (modelB) {  // Model-B conversion beta_B = beta_A/eps: the fluid carries the FULL -grad(p)
          Fx /= eP;    // (no -eps*grad p split), so the drag absorbs the particles' share. eps>=eps_min>0.
          Fy /= eP;
          Fz /= eP;
        }
        fdrag(p, 0) = (typename OutV::value_type)Fx;
        fdrag(p, 1) = (typename OutV::value_type)Fy;
        fdrag(p, 2) = (typename OutV::value_type)Fz;
        scatterAtMasked(fx, sdf, b, sx, sy, sz, wx, wy, wz, -Fx * invVcell);
        scatterAtMasked(fy, sdf, b, sx, sy, sz, wx, wy, wz, -Fy * invVcell);
        scatterAtMasked(fz, sdf, b, sx, sy, sz, wx, wy, wz, -Fz * invVcell);
      });
}

// Implicit-drag feedback (fused). Same as computeDragFeedback but instead of scattering the explicit
// reaction force it deposits the linear-drag COEFFICIENT density (beta_over_n * invVcell) onto
// `dragBeta` and the drag TARGET (beta_over_n * u_p * invVcell) onto (fx,fy,fz) — so the fluid solve
// treats -beta*(u - u_p) implicitly (unconditionally stable for a stiff bed). The particle drag
// force (evaluated at the current slip) still goes to `fdrag` (explicit on the particle side).
// beta_over_n = |F_p| / |vrel|, recovered from the drag law by a unit-slip evaluation.
template <class PosV, class VelV, class RadV, class FieldV, class OutV>
void computeDragImplicit(int np, PosV pos, VelV vel, RadV rad, FieldV uf, FieldV vf, FieldV wf,
                         FieldV eps, FieldV sdf, OutV fdrag, FieldV dragBeta, FieldV fx, FieldV fy,
                         FieldV fz, GridMap m, double mu, double rhof, double invVcell, int dragKind,
                         bool modelB) {
  using Exec = Kokkos::DefaultExecutionSpace;
  const int nx = m.ex - 2 * m.g, ny = m.ey - 2 * m.g, nz = m.ez - 2 * m.g;
  Kokkos::parallel_for(
      "peclet::coupling::drag_implicit", Kokkos::RangePolicy<Exec>(0, np), KOKKOS_LAMBDA(int p) {
        int i0, j0, k0;
        double wx, wy, wz;
        peclet::core::interp::detail::axisStencil((double)pos(p, 0), m.ox, m.idx, nx, i0, wx);
        peclet::core::interp::detail::axisStencil((double)pos(p, 1), m.oy, m.idy, ny, j0, wy);
        peclet::core::interp::detail::axisStencil((double)pos(p, 2), m.oz, m.idz, nz, k0, wz);
        const long sx = 1, sy = m.ex, sz = (long)m.ex * m.ey;
        const long b = (long)(i0 + m.g) + (long)(j0 + m.g) * sy + (long)(k0 + m.g) * sz;
        // Wall-aware gather: fluid corners only (partition of unity).
        const double uF = gatherAtMasked(uf, sdf, b, sx, sy, sz, wx, wy, wz, 0.0);
        const double vF = gatherAtMasked(vf, sdf, b, sx, sy, sz, wx, wy, wz, 0.0);
        const double wF = gatherAtMasked(wf, sdf, b, sx, sy, sz, wx, wy, wz, 0.0);
        const double eP = gatherAtMasked(eps, sdf, b, sx, sy, sz, wx, wy, wz, 1.0);
        const double upx = (double)vel(p, 0), upy = (double)vel(p, 1), upz = (double)vel(p, 2);
        const double vrx = uF - upx, vry = vF - upy, vrz = wF - upz;
        double Fx, Fy, Fz;
        dragForce(dragKind, vrx, vry, vrz, (double)rad(p), mu, rhof, eP, Fx, Fy, Fz);
        if (modelB) {  // Model-B conversion beta_B = beta_A/eps (fluid carries the full -grad p)
          Fx /= eP;
          Fy /= eP;
          Fz /= eP;
        }
        fdrag(p, 0) = (typename OutV::value_type)Fx;
        fdrag(p, 1) = (typename OutV::value_type)Fy;
        fdrag(p, 2) = (typename OutV::value_type)Fz;
        // beta_over_n = |F|/|vrel| (isotropic linear coefficient at the frozen slip)
        const double vmag = Kokkos::sqrt(vrx * vrx + vry * vry + vrz * vrz);
        const double Fmag = Kokkos::sqrt(Fx * Fx + Fy * Fy + Fz * Fz);
        const double bon = (vmag > 1e-30) ? Fmag / vmag : 0.0;
        scatterAtMasked(dragBeta, sdf, b, sx, sy, sz, wx, wy, wz, bon * invVcell);
        scatterAtMasked(fx, sdf, b, sx, sy, sz, wx, wy, wz, bon * upx * invVcell);
        scatterAtMasked(fy, sdf, b, sx, sy, sz, wx, wy, wz, bon * upy * invVcell);
        scatterAtMasked(fz, sdf, b, sx, sy, sz, wx, wy, wz, bon * upz * invVcell);
      });
}

}  // namespace peclet::coupling

#endif  // PECLET_COUPLING_COUPLING_KERNELS_HPP
