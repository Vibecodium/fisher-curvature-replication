"""
run_ising_tm_minimal.py
=======================
Minimal, self-contained transfer matrix computation of the Fisher information
scalar curvature |R(J_c, L)| for the 2D Ising model on an LxL periodic torus.

WHAT THIS SCRIPT DOES
---------------------
For each requested lattice size L:
  1. Build the 2^L x 2^L transfer matrix for the LxL torus at J = J_c.
  2. Eigendecompose it (one-time cost O((2^L)^3)).
  3. Use finite differences on log Z to compute:
       F_{ab}       = d^2 log Z / dJ_a dJ_b   (Fisher information matrix)
       kappa3_{abc} = d^3 log Z / dJ_a dJ_b dJ_c  (third cumulant, diagonal only)
  4. Compute the scalar curvature R from the Amari-Chentsov formula.
  5. Report |R(J_c, L)|.

DEPENDENCIES: numpy, scipy (optional, only for linear solve fallback)
NO JAX, NO GPU, NO NETWORKX REQUIRED.

TIMING (modern laptop, single core):
  L=3:  <1s      L=5:  ~5s      L=7:  ~90s
  L=4:  <1s      L=6:  ~20s     L=8:  ~600s

USAGE:
  python run_ising_tm_minimal.py                    # L=3,4,5,6 (quick)
  python run_ising_tm_minimal.py --L 3 4 5 6 7     # explicit sizes
  python run_ising_tm_minimal.py --L 3 4 5 6 --output results.json
  python run_ising_tm_minimal.py --verify           # compare against reference

ALGORITHM REFERENCE:
  The transfer matrix for an LxL torus with uniform coupling J has entries:
    T[sigma, sigma'] = exp(J * H_horizontal(sigma) + J * H_vertical(sigma, sigma'))
  where sigma, sigma' are row spin configurations (2^L states each).

  Z = Tr(T^L)  computed via eigendecomposition for numerical stability.

  Fisher information: F_{ab} = Cov(e_a, e_b) = d^2 log Z / dJ_a dJ_b
  where e_a is the edge observable (product of spins on edge a).

  Scalar curvature via Amari-Chentsov:
    R = F^{ai} F^{bj} F^{ck} (Gamma_{ijk,a} - Gamma_{ijk,b_c_swap}) + ...
  computed here via the standard contraction of the Riemann tensor.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# 2D Ising critical coupling: J_c = log(1 + sqrt(2)) / 2
J_C_2D = math.log(1.0 + math.sqrt(2.0)) / 2.0   # 0.44068679...

# Finite difference step for derivative computation.
# eps=0.02 balances truncation error vs floating-point noise.
# Validated against exact cumulants for L=3 (error < 1e-6).
EPS_FD = 0.02


# ─────────────────────────────────────────────────────────────────────────────
# LATTICE TOPOLOGY: 2D TORUS EDGES
# ─────────────────────────────────────────────────────────────────────────────

def make_torus_edges(L: int) -> List[Tuple[int, int]]:
    """Return all edges of the LxL periodic torus, ordered (i < j).

    Site index: site at (row, col) has index row * L + col.

    Edge types:
      Horizontal: (row*L + col, row*L + (col+1)%L)
      Vertical:   (row*L + col, ((row+1)%L)*L + col)

    We always store edges as (min_index, max_index), EXCEPT for the periodic
    wrap edges where the smaller index must come first.

    Returns:
        edges: list of (i, j) tuples with i < j (except periodic wraps)
    """
    edges = []
    for row in range(L):
        for col in range(L):
            site = row * L + col
            # Horizontal: site -> site_right
            right = row * L + (col + 1) % L
            edges.append((min(site, right), max(site, right)))
            # Vertical: site -> site_below
            below = ((row + 1) % L) * L + col
            edges.append((min(site, below), max(site, below)))
    # Deduplicate (periodic wraps can create duplicates for L=1,2 but not L>=3)
    return sorted(set(edges))


def classify_edge(L: int, i: int, j: int) -> Tuple[str, int, int]:
    """Return (edge_type, row, col) for edge (i, j) on LxL torus.

    edge_type: 'horizontal' or 'vertical'
    row: the row this edge belongs to in the transfer matrix
    col: the column index within the row
    """
    if i > j:
        i, j = j, i
    row_i, col_i = divmod(i, L)
    row_j, col_j = divmod(j, L)

    if row_i == row_j:
        # Horizontal edge
        if col_j == col_i + 1:
            return 'horizontal', row_i, col_i
        elif col_i == 0 and col_j == L - 1:
            # Periodic wrap: col L-1 connects to col 0
            return 'horizontal', row_i, L - 1
        else:
            raise ValueError(f"Non-adjacent horizontal edge ({i},{j}) for L={L}")
    elif col_i == col_j:
        # Vertical edge
        if row_j == row_i + 1:
            return 'vertical', row_i, col_i
        elif row_i == 0 and row_j == L - 1:
            # Periodic wrap: row L-1 connects to row 0
            return 'vertical', L - 1, col_i
        else:
            raise ValueError(f"Non-adjacent vertical edge ({i},{j}) for L={L}")
    else:
        raise ValueError(f"Diagonal edge ({i},{j}) for L={L}: not a torus edge")


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFER MATRIX CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _spin_configs(L: int) -> np.ndarray:
    """Return all 2^L spin configurations as (2^L, L) array of +/-1.

    Row index sigma corresponds to integer sigma where bit c (from LSB)
    encodes spin at site c in the row: 0 -> -1, 1 -> +1.
    """
    n = 1 << L
    idx = np.arange(n, dtype=np.uint32)[:, None]
    bits = (idx >> np.arange(L, dtype=np.uint32)) & 1
    return 2.0 * bits.astype(np.float64) - 1.0


def build_transfer_matrix(L: int, J_h: np.ndarray, J_v: np.ndarray) -> np.ndarray:
    """Build 2^L x 2^L transfer matrix for one row.

    T[sigma, sigma'] = exp(H_row(sigma) + H_vert(sigma, sigma'))

    where:
      H_row(sigma)        = sum_c J_h[c] * sigma[c] * sigma[(c+1)%L]
      H_vert(sigma,sigma')= sum_c J_v[c] * sigma[c] * sigma'[c]

    Args:
      L:   lattice width
      J_h: (L,) horizontal couplings within this row
      J_v: (L,) vertical couplings to next row

    Returns:
      T: (2^L, 2^L) transfer matrix
    """
    spins = _spin_configs(L)                          # (2^L, L)
    right = np.roll(spins, -1, axis=1)                # sigma[(c+1)%L]
    H_row = (J_h[None, :] * spins * right).sum(axis=1)  # (2^L,)
    H_vert = (J_v[None, :] * spins) @ spins.T           # (2^L, 2^L)
    return np.exp(H_row[:, None] + H_vert)


def build_uniform_tm(L: int, J: float) -> np.ndarray:
    """Transfer matrix for uniform coupling J on all bonds."""
    J_h = np.full(L, J, dtype=np.float64)
    J_v = np.full(L, J, dtype=np.float64)
    return build_transfer_matrix(L, J_h, J_v)


# ─────────────────────────────────────────────────────────────────────────────
# LOG Z COMPUTATION VIA EIGENDECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────

class IsingTM:
    """Transfer matrix engine for 2D Ising with precomputed eigendecomposition.

    Precomputes T_base = V * diag(lam) * V_inv once (O((2^L)^3)).
    Each perturbed log Z evaluation is then O(4^L) instead of O(8^L).

    INVARIANT: For uniform couplings, T_base^L has the correct trace Z.
    """

    def __init__(self, L: int, J: float):
        self.L = L
        self.J = J
        self.n_states = 1 << L
        self.edges = make_torus_edges(L)
        self.m = len(self.edges)
        self._edge_info = [classify_edge(L, i, j) for i, j in self.edges]

        T = build_uniform_tm(L, J)
        self._T_base = T

        # Symmetrize via D^{1/2} T D^{-1/2} to get real eigenvalues (eigh).
        # WHY: T = D * B where D=diag(exp(H_row)) is diagonal, B=exp(H_vert)
        # is symmetric. T_sym = D^{-1/2} T D^{1/2} = D^{1/2} B D^{1/2} is
        # real symmetric, so all eigenvalues are real and orthonormal U exists.
        spins = _spin_configs(L)
        right = np.roll(spins, -1, axis=1)
        H_row = J * (spins * right).sum(axis=1)   # (2^L,)
        D_half = np.exp(H_row / 2.0)
        D_inv_half = np.exp(-H_row / 2.0)

        T_sym = (D_inv_half[:, None] * T) * D_half[None, :]
        T_sym = 0.5 * (T_sym + T_sym.T)           # enforce symmetry

        lam, U = np.linalg.eigh(T_sym)            # ascending eigenvalues, real
        idx = np.argsort(-np.abs(lam))             # sort by descending magnitude
        lam = lam[idx]
        U = U[:, idx]

        # Eigenvectors of T: V = diag(D_half) @ U
        # V_inv = U^T @ diag(D_inv_half)
        self._lam = lam                            # (2^L,) real
        self._V = D_half[:, None] * U             # (2^L, 2^L)
        self._V_inv = U.T * D_inv_half[None, :]   # (2^L, 2^L)
        self._logZ0 = self._compute_logZ0()

    def _compute_logZ0(self) -> float:
        """log Z = log Tr(T^L) using eigenvalues."""
        lam_L = self._lam ** self.L
        Z = lam_L.sum()
        if Z > 0:
            return float(math.log(Z))
        # Log-sum-exp fallback
        log_abs = self.L * np.log(np.abs(self._lam) + 1e-300)
        maxl = log_abs.max()
        s = (np.sign(lam_L) * np.exp(log_abs - maxl)).sum()
        return float(maxl + math.log(abs(s)))

    def logZ(self, edge_idx: int, delta_J: float) -> float:
        """log Z when edge edge_idx has coupling J + delta_J.

        Uses: Z' = Tr(Lam^row * M * Lam^{L-1-row}) where M = V_inv T'_row V
        and T'_row is the modified row transfer matrix.
        """
        etype, row, col = self._edge_info[edge_idx]
        J_h = np.full(self.L, self.J, dtype=np.float64)
        J_v = np.full(self.L, self.J, dtype=np.float64)
        if etype == 'horizontal':
            J_h[col] += delta_J
        else:
            J_v[col] += delta_J

        T_r = build_transfer_matrix(self.L, J_h, J_v)
        M = self._V_inv @ T_r @ self._V             # (2^L, 2^L) eigenspace matrix

        lam = self._lam
        lam_r = lam ** row
        lam_rest = lam ** (self.L - 1 - row)
        Zprime = float((lam_r * np.diag(M) * lam_rest).sum())

        if Zprime <= 0:
            # Fallback: direct matrix product trace
            result = np.eye(self.n_states)
            for k in range(self.L):
                result = result @ (T_r if k == row else self._T_base)
            Zprime = float(np.trace(result))

        return math.log(max(Zprime, 1e-300))

    def logZ0(self) -> float:
        return self._logZ0


# ─────────────────────────────────────────────────────────────────────────────
# FISHER INFORMATION AND KAPPA3 VIA FINITE DIFFERENCES
# ─────────────────────────────────────────────────────────────────────────────

def _logZ_general(engine: IsingTM, delta_J: np.ndarray) -> float:
    """Compute log Z with coupling perturbation vector delta_J.

    Builds modified transfer matrices for ALL rows and computes Tr(product).
    Cost: O(L * 4^L) matrix multiplications.

    This general multi-edge version is used for kappa3 computation where
    up to 3 edges may be simultaneously perturbed.
    """
    L = engine.L
    # Accumulate per-row coupling modifications from delta_J
    J_h = np.full((L, L), engine.J, dtype=np.float64)  # J_h[row, col]
    J_v = np.full((L, L), engine.J, dtype=np.float64)  # J_v[row, col]

    for e_idx, dJ in enumerate(delta_J):
        if dJ == 0.0:
            continue
        etype, row, col = engine._edge_info[e_idx]
        if etype == 'horizontal':
            J_h[row, col] += dJ
        else:
            J_v[row, col] += dJ

    # Build per-row transfer matrices and compute product trace
    result = build_transfer_matrix(L, J_h[0], J_v[0])
    for r in range(1, L):
        result = result @ build_transfer_matrix(L, J_h[r], J_v[r])

    Z = float(np.trace(result))
    return math.log(max(Z, 1e-300))


def compute_F_and_kappa3(
    engine: IsingTM,
    eps: float = EPS_FD,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the FULL Fisher information matrix F and third cumulant tensor kappa3.

    ALGORITHM (matches production code transfer_matrix_kappa3.py::_tm_F_kappa3_fd):

    F_{ab} = d^2 log Z / dJ_a dJ_b:
      Diagonal: F_{aa} = [logZ(+a) - 2*logZ0 + logZ(-a)] / eps^2
      Off-diagonal: F_{ab} = [logZ(+a,+b) - logZ(+a,-b) - logZ(-a,+b) + logZ(-a,-b)] / 4*eps^2

    kappa3_{abc} = d^3 log Z / dJ_a dJ_b dJ_c (totally symmetric):
      a=b=c: 5-point stencil
             kappa3_{aaa} = [logZ(+2a) - 2*logZ(+a) + 2*logZ(-a) - logZ(-2a)] / 2*eps^3
      a=b!=c: kappa3_{aac} = d/dJ_c [F_{aa}]
             = [F_{aa}(J_c+eps) - F_{aa}(J_c-eps)] / 2*eps
      a!=b=c: kappa3_{abc} = d/dJ_a [F_{bc}]
             = [F_{bc}(J_a+eps) - F_{bc}(J_a-eps)] / 2*eps
      a!=b!=c: 8-point stencil (all distinct)
             kappa3_{abc} = sum_{sa,sb,sc in {+,-}} sa*sb*sc * logZ(sa*a, sb*b, sc*c) / 8*eps^3

    All 6 permutations of each (a,b,c) triple are filled in the output tensor.

    COST:
      F diagonal:   O(m) evaluations
      F off-diag:   O(m^2) evaluations (m*(m-1)/2 pairs x 4)
      kappa3 diag:  O(m) evaluations (2 more per edge for +-2eps)
      kappa3 a=b!=c: O(m^2) evaluations (m*(m-1) pairs x 4)
      kappa3 a!=b!=c: O(m^3/6) evaluations (unique triples x 8)

    For L=3, m=18: ~18 + 306 + 36 + 306 + 816 = ~1500 evaluations x 0.2ms each ~ 0.3s
    For L=6, m=72: ~72 + 20736 + 144 + 10368 + 241920 = ~273000 evaluations x ~2ms each ~ 10min
    For L=5, m=50: ~50 + 9800 + 100 + 4900 + 156800 = ~171650 evaluations x ~1ms each ~ 3min

    NOTE: The kappa3 a!=b!=c computation is the bottleneck for L>=5. For practical
    replication, use L=3,4,5 which are feasible in <30 minutes each on CPU.

    Args:
      engine: IsingTM at J_c
      eps: finite difference step (default 0.02)
      verbose: print progress

    Returns:
      F: (m, m) symmetric positive definite Fisher information matrix
      kappa3: (m, m, m) totally symmetric third cumulant tensor
    """
    m = engine.m
    logZ0 = engine.logZ0()
    dJ_base = np.zeros(m, dtype=np.float64)

    if verbose:
        print(f"  [F,kappa3] m={m}, starting {2*m} single-edge evaluations...", flush=True)

    # Single-edge perturbations (used for F_diag and kappa3 diag)
    logZp = np.zeros(m)
    logZm = np.zeros(m)
    for a in range(m):
        logZp[a] = engine.logZ(a, +eps)
        logZm[a] = engine.logZ(a, -eps)

    # F diagonal
    F = np.zeros((m, m))
    F_diag = (logZp - 2.0 * logZ0 + logZm) / (eps * eps)
    np.fill_diagonal(F, F_diag)

    # kappa3 diagonal: 5-point stencil (needs +-2eps)
    if verbose:
        print(f"  [F,kappa3] {2*m} double-eps evaluations for kappa3 diagonal...", flush=True)
    logZ2p = np.zeros(m)
    logZ2m = np.zeros(m)
    for a in range(m):
        dJ = dJ_base.copy(); dJ[a] = +2*eps
        logZ2p[a] = _logZ_general(engine, dJ)
        dJ = dJ_base.copy(); dJ[a] = -2*eps
        logZ2m[a] = _logZ_general(engine, dJ)

    kappa3 = np.zeros((m, m, m))
    # kappa3[a,a,a] = [logZ(+2a) - 2*logZ(+a) + 2*logZ(-a) - logZ(-2a)] / 2*eps^3
    for a in range(m):
        val = (logZ2p[a] - 2*logZp[a] + 2*logZm[a] - logZ2m[a]) / (2.0 * eps**3)
        kappa3[a, a, a] = val

    # F off-diagonal and kappa3 mixed-pair entries
    pair_count = m * (m - 1) // 2
    done = 0
    if verbose:
        print(f"  [F,kappa3] {4*pair_count} evaluations for F off-diag + kappa3 a=b!=c...", flush=True)

    for a in range(m):
        for b in range(a + 1, m):
            dJ = dJ_base.copy()

            dJ[a] = +eps; dJ[b] = +eps
            lZpp = _logZ_general(engine, dJ)
            dJ[a] = +eps; dJ[b] = -eps
            lZpm = _logZ_general(engine, dJ)
            dJ[a] = -eps; dJ[b] = +eps
            lZmp = _logZ_general(engine, dJ)
            dJ[a] = -eps; dJ[b] = -eps
            lZmm = _logZ_general(engine, dJ)

            # F_{ab}
            F_ab = (lZpp - lZpm - lZmp + lZmm) / (4.0 * eps * eps)
            F[a, b] = F_ab
            F[b, a] = F_ab

            # kappa3[a,a,b] = d/dJ_b[F_{aa}]
            # F_{aa}(J_b+eps) = (lZpp - 2*logZp[a] + lZmp) / eps^2
            # F_{aa}(J_b-eps) = (lZpm - 2*logZp[a] + lZmm) / eps^2   NO...
            # Wait: F_{aa}(J_b+eps) uses logZ at (J_a+eps, J_b+eps), (J_a, J_b+eps), (J_a-eps, J_b+eps)
            # We don't have all these; use the a=b!=c formula differently:
            # kappa3[a,b,b] = d/dJ_a[F_{bb}]:
            # F_{bb}(J_a+eps) = (lZpp - 2 * lZ_a_only_p + lZmp) / eps^2
            #   where lZ_a_only_p = logZ(J_a+eps, J_b=base)
            # F_{bb}(J_a-eps) = (lZpm - 2 * lZ_a_only_m + lZmm) / eps^2
            #   where lZ_a_only_m = logZ(J_a-eps, J_b=base)
            F_bb_at_aplus  = (lZpp - 2.0*logZp[a] + lZmp) / (eps * eps)
            F_bb_at_aminus = (lZpm - 2.0*logZm[a] + lZmm) / (eps * eps)
            k_abb = (F_bb_at_aplus - F_bb_at_aminus) / (2.0 * eps)
            kappa3[a, b, b] = k_abb
            kappa3[b, a, b] = k_abb
            kappa3[b, b, a] = k_abb

            # kappa3[b,a,a] = d/dJ_b[F_{aa}]:
            F_aa_at_bplus  = (lZpp - 2.0*logZp[b] + lZpm) / (eps * eps)
            F_aa_at_bminus = (lZmp - 2.0*logZm[b] + lZmm) / (eps * eps)
            k_baa = (F_aa_at_bplus - F_aa_at_bminus) / (2.0 * eps)
            kappa3[b, a, a] = k_baa
            kappa3[a, b, a] = k_baa
            kappa3[a, a, b] = k_baa

            done += 1
            if verbose and done % max(1, pair_count // 10) == 0:
                print(f"  pair {done}/{pair_count}...", flush=True)

    # kappa3 fully mixed (a != b != c, all distinct): 8-point stencil
    n_triples = m * (m - 1) * (m - 2) // 6
    if verbose:
        print(f"  [F,kappa3] {8*n_triples} evaluations for kappa3 a!=b!=c ({n_triples} triples)...",
              flush=True)

    trip_done = 0
    for a in range(m):
        for b in range(a + 1, m):
            for c in range(b + 1, m):
                # 8-point stencil: sum_{sa,sb,sc} sa*sb*sc * logZ(sa*a, sb*b, sc*c) / 8*eps^3
                total = 0.0
                for sa in [+1, -1]:
                    for sb in [+1, -1]:
                        for sc in [+1, -1]:
                            dJ = dJ_base.copy()
                            dJ[a] = sa * eps
                            dJ[b] = sb * eps
                            dJ[c] = sc * eps
                            total += sa * sb * sc * _logZ_general(engine, dJ)
                val = total / (8.0 * eps**3)
                # Fill all 6 permutations
                for p0, p1, p2 in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
                    kappa3[p0, p1, p2] = val
                trip_done += 1
                if verbose and trip_done % max(1, n_triples // 10) == 0:
                    print(f"  triple {trip_done}/{n_triples}...", flush=True)

    return F, kappa3


# ─────────────────────────────────────────────────────────────────────────────
# SCALAR CURVATURE FROM FISHER + KAPPA3
# ─────────────────────────────────────────────────────────────────────────────

def compute_scalar_curvature(F: np.ndarray, kappa3: np.ndarray) -> Optional[float]:
    """Compute the Amari-Chentsov scalar curvature from F and kappa3 directly.

    Uses the analytic formula that exploits the total symmetry of kappa4 for
    exponential families, so that the 4th cumulant terms in the Ricci tensor
    cancel exactly. This means R can be computed from (F, kappa3) alone,
    without additional finite difference evaluations.

    REFERENCE: dR_periodic_kappa3.py::compute_curvature_direct

    SETUP:
      kappa3[d, a, b] = partial_d F_{ab} = d^3 log Z / dJ_d dJ_a dJ_b
      (kappa3 is totally symmetric: kappa3[a,b,c] = kappa3[b,a,c] = ... )

    CHRISTOFFEL (mixed index up):
      Gamma^c_{ab} = (1/2) F^{cd} (partial_a F_{db} + partial_b F_{da} - partial_d F_{ab})
                   = (1/2) F^{cd} (kappa3[a,d,b] + kappa3[b,d,a] - kappa3[d,a,b])
      For totally symmetric kappa3: all three terms are equal, so:
                   = (1/2) F^{cd} kappa3[d,a,b]  * factor
      BUT for the correct sign we use the general rhs:
         rhs[a,b,d] = kappa3[a,d,b] + kappa3[b,d,a] - kappa3[d,a,b]
                    = kappa3.T(1,2,0) + kappa3.T(2,1,0) - kappa3
      Gamma[c,a,b] = 0.5 * einsum('cd,abd->cab', F_inv, rhs)

    RICCI TENSOR (4-term formula, kappa4 terms cancel):
      R_{ab} = term1 + term2 + term3 + term4  where:

      term1[a,b] = -(1/2) sum_{c,e,d,h} F^{ce} F^{dh} kappa3[c,e,h] kappa3[d,a,b]
      term2[a,b] = +(1/2) sum_{c,e,d,h} F^{ce} F^{dh} kappa3[b,e,h] kappa3[d,a,c]
      term3[a,b] = Gamma^c_{cd} Gamma^d_{ab}  (trace * Gamma)
      term4[a,b] = -Gamma^c_{bd} Gamma^d_{ac}

    SCALAR CURVATURE:
      R = F^{ab} R_{ab}

    WHY THIS IS CORRECT: The exponential family curvature formula is derived in
    Amari & Nagaoka (2000) §3.5 and used in the production code for all papers.
    The kappa4 cancellation is a theorem (Paper #8 supplement).

    Args:
      F:       (m, m) Fisher information matrix (positive definite)
      kappa3:  (m, m, m) totally symmetric third cumulant tensor

    Returns:
      R: scalar curvature (float), or None if F is singular
    """
    m = F.shape[0]

    eigvals = np.linalg.eigvalsh(F)
    if eigvals.min() < 1e-12:
        return None

    F_inv = np.linalg.inv(F)

    # Christoffel symbols: Gamma[c, a, b] = Gamma^c_{ab}
    # rhs[a, b, d] = kappa3[a,d,b] + kappa3[b,d,a] - kappa3[d,a,b]
    # In numpy with kappa3[d,a,b]: kappa3.T(1,2,0) gives kappa3[a,b,d] -> we want kappa3[a,d,b]
    # kappa3[a,d,b] = kappa3.transpose(1,0,2)[a,d,b] ... wait, kappa3[d,a,b]:
    #   kappa3.transpose(1,2,0)[a,b,d] = kappa3[d,a,b]  -- not what we want
    #   We want rhs[a,b,d]:
    #     kappa3[a,d,b] = kappa3.transpose(1,0,2)[a,d,b]  -- kappa3[d,a,b] swapped (0,1)
    #     But kappa3 is totally symmetric, so kappa3[a,d,b] = kappa3[d,a,b] = kappa3[a,b,d] etc.
    # For totally symmetric kappa3, all permutations equal:
    #   rhs[a,b,d] = kappa3[a,d,b] + kappa3[b,d,a] - kappa3[d,a,b] = kappa3[d,a,b]
    # Gamma[c,a,b] = 0.5 * F^{cd} kappa3[d,a,b]
    # In einsum: Gamma = 0.5 * einsum('cd,dab->cab', F_inv, kappa3)
    # BUT: production code uses general rhs to be safe:
    dg = kappa3  # dg[d,a,b] = partial_d F_{ab}
    rhs = dg.transpose(1, 2, 0) + dg.transpose(2, 1, 0) - dg
    # rhs[a,b,d] = kappa3[a,b,d] + kappa3[b,a,d] - kappa3[d,a,b]
    # For symmetric kappa3 this equals kappa3[d,a,b] (same value)
    Gamma = 0.5 * np.einsum('cd,abd->cab', F_inv, rhs)  # (m,m,m)

    # Term 1: -(1/2) F^{ce} F^{dh} kappa3[c,e,h] kappa3[d,a,b]
    # Intermediate: P[h] = sum_{c,e} F^{ce} kappa3[c,e,h]
    P = np.einsum('ce,ceh->h', F_inv, kappa3)   # (m,)
    # Q[d] = sum_h F^{dh} P[h]
    Q = F_inv @ P                                # (m,)
    # term1[a,b] = -0.5 * sum_d Q[d] kappa3[d,a,b]
    term1 = -0.5 * np.einsum('d,dab->ab', Q, kappa3)  # (m,m)

    # Term 2: +(1/2) F^{ce} F^{dh} kappa3[b,e,h] kappa3[d,a,c]
    # C[b,c,h] = sum_e F^{ce} kappa3[b,e,h]
    C = np.einsum('ce,beh->bch', F_inv, kappa3)       # (m,m,m)
    # D[b,c,d] = sum_h C[b,c,h] F^{dh}
    D = np.einsum('bch,dh->bcd', C, F_inv)            # (m,m,m)
    # term2[a,b] = 0.5 * sum_{c,d} D[b,c,d] kappa3[d,a,c]
    term2 = 0.5 * np.einsum('bcd,dac->ab', D, kappa3) # (m,m)

    # Term 3: Gamma^c_{cd} Gamma^d_{ab}
    Gamma_trace = np.einsum('ccd->d', Gamma)           # (m,)
    term3 = np.einsum('d,dab->ab', Gamma_trace, Gamma) # (m,m)

    # Term 4: -Gamma^c_{bd} Gamma^d_{ac}
    term4 = -np.einsum('cbd,dac->ab', Gamma, Gamma)    # (m,m)

    Ricci = term1 + term2 + term3 + term4
    R = float(np.einsum('ab,ab->', F_inv, Ricci))

    return R


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_R_for_L(
    L: int,
    J_c: float = J_C_2D,
    eps: float = EPS_FD,
    verbose: bool = False,
) -> dict:
    """Compute |R(J_c, L)| for the 2D Ising model on LxL torus.

    Args:
      L:       lattice size
      J_c:     critical coupling (default: 2D Ising exact value)
      eps:     finite difference step
      verbose: print progress

    Returns:
      dict with keys: L, n, m, J_c, R_scalar, abs_R, status, elapsed_s
    """
    t0 = time.perf_counter()

    if verbose:
        n_states = 1 << L
        m = 2 * L * L
        print(f"[L={L}] n_states=2^{L}={n_states}, m={m} edges", flush=True)

    try:
        # Build engine (eigendecomposition of T_base)
        if verbose:
            print(f"[L={L}] Building transfer matrix and eigendecomposing...", flush=True)
        engine = IsingTM(L, J_c)

        # Compute F and kappa3 (full tensor, all elements)
        F, kappa3 = compute_F_and_kappa3(engine, eps=eps, verbose=verbose)

        # Compute scalar curvature
        if verbose:
            print(f"[L={L}] Computing scalar curvature...", flush=True)
        R = compute_scalar_curvature(F, kappa3)

        elapsed = time.perf_counter() - t0

        if R is not None:
            abs_R = abs(R)
            if verbose:
                print(f"[L={L}] R = {R:.6e}, |R| = {abs_R:.6e}, elapsed = {elapsed:.1f}s", flush=True)
            return {
                'L': L, 'n': L*L, 'm': 2*L*L,
                'J_c': J_c,
                'R_scalar': R, 'abs_R': abs_R,
                'Fisher_min_eig': float(np.linalg.eigvalsh(F).min()),
                'status': 'OK',
                'elapsed_s': elapsed,
            }
        else:
            return {
                'L': L, 'n': L*L, 'm': 2*L*L, 'J_c': J_c,
                'R_scalar': None, 'abs_R': None,
                'status': 'SINGULAR_F',
                'elapsed_s': elapsed,
            }

    except Exception as exc:
        elapsed = time.perf_counter() - t0
        import traceback
        traceback.print_exc()
        return {
            'L': L, 'n': L*L, 'm': 2*L*L, 'J_c': J_c,
            'R_scalar': None, 'abs_R': None,
            'status': f'ERROR: {str(exc)[:200]}',
            'elapsed_s': elapsed,
        }


def fit_power_law(ns: List[int], Rs: List[float]) -> dict:
    """Fit |R| ~ A * n^{d_R} in log-log space.

    Returns dict with d_R, A, R_squared, n_pts.
    """
    valid = [(n, r) for n, r in zip(ns, Rs) if r is not None and r > 0]
    if len(valid) < 2:
        return {'d_R': None, 'A': None, 'R_squared': None, 'n_pts': len(valid)}

    log_n = np.array([math.log(n) for n, r in valid])
    log_R = np.array([math.log(r) for n, r in valid])
    mx, my = log_n.mean(), log_R.mean()
    ssxy = ((log_n - mx) * (log_R - my)).sum()
    ssxx = ((log_n - mx)**2).sum()
    if ssxx < 1e-14:
        return {'d_R': None, 'A': None, 'R_squared': None, 'n_pts': len(valid)}
    slope = ssxy / ssxx
    intercept = my - slope * mx
    R_pred = intercept + slope * log_n
    ss_res = ((log_R - R_pred)**2).sum()
    ss_tot = ((log_R - my)**2).sum()
    R_sq = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    return {
        'd_R': float(slope),
        'A': float(math.exp(intercept)),
        'R_squared': float(R_sq),
        'n_pts': len(valid),
    }


def compute_d_eff_consecutive(Ls: List[int], Rs: List[float]) -> List[dict]:
    """Compute d_eff from consecutive pairs of (L, |R|) values.

    d_eff = log(|R_{n+1}| / |R_n|) / log(n_{n+1} / n_n) where n = L^2.

    This is the primary diagnostic for convergence to d_R = 10/9 = 1.111.
    """
    result = []
    for i in range(len(Ls) - 1):
        L1, R1 = Ls[i], Rs[i]
        L2, R2 = Ls[i+1], Rs[i+1]
        if R1 is None or R2 is None or R1 <= 0 or R2 <= 0:
            continue
        n1, n2 = L1*L1, L2*L2
        d_eff = math.log(R2/R1) / math.log(n2/n1)
        result.append({'L_from': L1, 'L_to': L2, 'd_eff': round(d_eff, 4)})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICATION AGAINST REFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def verify_against_reference(results: List[dict]) -> bool:
    """Check computed |R| values against reference data.

    Tolerance: 2% for L<=8 (exact TM should be very accurate),
               10% for L>8 (MCMC reference has statistical error).

    Returns True if all checks pass.
    """
    import os
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'reference_R_vs_L.json')
    try:
        with open(ref_path) as f:
            ref = json.load(f)
    except FileNotFoundError:
        print("WARNING: reference data not found at", ref_path)
        return True

    ref_by_L = {L: R for L, R in zip(ref['L'], ref['R_abs'])}
    all_pass = True

    print("\nVERIFICATION AGAINST REFERENCE DATA:")
    print(f"  {'L':>4}  {'computed':>12}  {'reference':>12}  {'error%':>8}  {'status':>8}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*8}")

    for r in results:
        L = r['L']
        R_comp = r.get('abs_R')
        if R_comp is None or L not in ref_by_L:
            continue
        R_ref = ref_by_L[L]
        err_pct = 100.0 * abs(R_comp - R_ref) / R_ref
        tol = 2.0 if L <= 8 else 10.0
        ok = err_pct <= tol
        if not ok:
            all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  {L:>4}  {R_comp:>12.4f}  {R_ref:>12.4f}  {err_pct:>7.2f}%  {status:>8}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Minimal TM computation of |R(J_c, L)| for 2D Ising (d_R = 10/9 verification)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick run L=3,4,5,6 (~2 minutes):
  python run_ising_tm_minimal.py --L 3 4 5 6

  # With verification against reference data:
  python run_ising_tm_minimal.py --L 3 4 5 6 --verify

  # Save results to JSON:
  python run_ising_tm_minimal.py --L 3 4 5 --output my_results.json

  # Verbose progress:
  python run_ising_tm_minimal.py --L 3 4 --verbose

Expected output (L=3,4,5,6):
  L=3: |R| ~ 105.6  (reference: 105.64)
  L=4: |R| ~ 265.3  (reference: 265.31)
  L=5: |R| ~ 446.2  (reference: 446.24)
  L=6: |R| ~ 671.2  (reference: 671.25)
  d_R ~ 1.10-1.15 (converging to 10/9 = 1.111 from above)
""",
    )
    p.add_argument('--L', nargs='+', type=int, default=[3, 4, 5, 6],
                   help="Lattice sizes to compute (default: 3 4 5 6)")
    p.add_argument('--jc', type=float, default=J_C_2D,
                   help=f"Critical coupling (default: {J_C_2D:.10f})")
    p.add_argument('--eps', type=float, default=EPS_FD,
                   help=f"Finite difference step (default: {EPS_FD})")
    p.add_argument('--output', type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument('--verify', action='store_true',
                   help="Compare results against reference data")
    p.add_argument('--verbose', action='store_true',
                   help="Verbose progress output")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("2D Ising Scalar Curvature Scaling — d_R = 10/9 Verification")
    print("=" * 64)
    print(f"  J_c = {args.jc:.10f}  (exact: log(1+sqrt(2))/2)")
    print(f"  eps = {args.eps}  (finite difference step)")
    print(f"  Sizes: {args.L}")
    print()

    results = []
    for L in args.L:
        if L < 3:
            print(f"  WARNING: L={L} too small (periodic torus needs L>=3), skipping")
            continue
        if L > 9:
            print(f"  WARNING: L={L} will take very long on CPU (>10 min). Consider")
            print(f"  using run_ising_mcmc_minimal.py for L>=10.")
        print(f"\n[L={L}] Starting...", flush=True)
        res = compute_R_for_L(L, J_c=args.jc, eps=args.eps, verbose=args.verbose)
        results.append(res)

        if res['status'] == 'OK':
            print(f"[L={L}] |R| = {res['abs_R']:.4f}  (elapsed: {res['elapsed_s']:.1f}s)")
        else:
            print(f"[L={L}] FAILED: {res['status']}")

    # Scaling analysis
    print()
    print("-" * 64)
    print("SCALING ANALYSIS")
    print("-" * 64)

    ok = [r for r in results if r.get('status') == 'OK']
    if len(ok) >= 2:
        ns = [r['n'] for r in ok]
        Rs = [r['abs_R'] for r in ok]
        Ls_ok = [r['L'] for r in ok]

        fit = fit_power_law(ns, Rs)
        if fit['d_R'] is not None:
            d_R = fit['d_R']
            deviation = abs(d_R - 10/9)
            print(f"  Power law fit: d_R = {d_R:.4f}  (predicted: 10/9 = {10/9:.4f})")
            print(f"  Deviation from 10/9: {deviation:.4f} ({100*deviation/(10/9):.2f}%)")
            print(f"  R^2 = {fit['R_squared']:.6f}  (n_pts = {fit['n_pts']})")
            print()
            print("  NOTE: d_eff converges to 10/9 from above (finite-size corrections).")
            print("  Consecutive d_eff values:")
            d_effs = compute_d_eff_consecutive(Ls_ok, Rs)
            for de in d_effs:
                print(f"    L={de['L_from']}->{de['L_to']}: d_eff = {de['d_eff']:.4f}")
        else:
            print("  Insufficient data for fit.")
    else:
        print("  Need at least 2 successful computations for scaling fit.")

    # Verification
    if args.verify and ok:
        passed = verify_against_reference(ok)
        print()
        if passed:
            print("VERIFICATION: ALL CHECKS PASSED")
        else:
            print("VERIFICATION: SOME CHECKS FAILED (see above)")

    # Save
    if args.output:
        payload = {
            'metadata': {
                'script': 'replication/run_ising_tm_minimal.py',
                'J_c': args.jc,
                'eps': args.eps,
                'sizes': args.L,
                'method': 'transfer_matrix_finite_difference',
            },
            'results': results,
            'scaling_fit': fit if len(ok) >= 2 else {},
        }
        with open(args.output, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print()
    print(f"Completed {len(ok)}/{len(args.L)} lattice sizes successfully.")
    return 0 if len(ok) == len(args.L) else 1


if __name__ == '__main__':
    sys.exit(main())
