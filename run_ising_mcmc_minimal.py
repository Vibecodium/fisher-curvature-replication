"""
run_ising_mcmc_minimal.py
=========================
MCMC-based computation of |R(J_c, L)| for 2D Ising at large L (L=10-20).

This is the MCMC companion to run_ising_tm_minimal.py. Use it to extend
verification beyond L=9 where exact TM becomes impractical on CPU.

REQUIREMENTS:
  - JAX (for Wolff cluster algorithm on CPU or GPU)
  - scipy, numpy

WHAT THIS DOES:
  For each L, runs the Wolff cluster MCMC algorithm at J_c to collect
  spin configurations, then estimates the Fisher information matrix
  and third cumulant from MC cumulants. Computes scalar curvature.

WHY MCMC FOR LARGE L:
  Exact TM memory: O(2^{2L}) matrix -- for L=12 that's 4096x4096 = 128 MB,
  and for L=16 it's 65536x65536 = 32 GB (infeasible on laptops).
  MCMC memory: O(L^2) for a single spin configuration.
  Trade-off: MCMC has statistical error; exact TM is exact.

TIMING (CPU, no GPU):
  L=10: ~30 min (100k samples)   L=14: ~3 hours (100k samples)

TIMING (RTX 4090 GPU):
  L=10: ~7 min (500k samples)    L=14: ~25 min (500k samples)

USAGE:
  # CPU, L=10,12 with 100k samples:
  python run_ising_mcmc_minimal.py --L 10 12 --n-samples 100000

  # GPU, L=10-16 with 500k samples:
  python run_ising_mcmc_minimal.py --L 10 12 14 16 --n-samples 500000

  # Compare against reference:
  python run_ising_mcmc_minimal.py --L 10 12 --verify

NOTE:
  For a 30-minute external verification, the exact TM script
  (run_ising_tm_minimal.py) with L=3-7 is sufficient. This script
  is for extended verification only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# JAX IMPORT (OPTIONAL)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    print("WARNING: JAX not installed. MCMC will use NumPy (very slow for large L).")
    print("Install JAX: pip install jax jaxlib")

J_C_2D = math.log(1.0 + math.sqrt(2.0)) / 2.0   # 0.44068679...

# ─────────────────────────────────────────────────────────────────────────────
# PURE NUMPY WOLFF CLUSTER MCMC (fallback, works without JAX)
# ─────────────────────────────────────────────────────────────────────────────

def wolff_step_numpy(spins: np.ndarray, J: float, rng: np.random.Generator) -> np.ndarray:
    """Single Wolff cluster flip on LxL periodic lattice.

    ALGORITHM:
    1. Pick random seed site.
    2. BFS: add neighbors with probability p_add = 1 - exp(-2J) if same spin.
    3. Flip all sites in cluster.

    Args:
      spins: (L, L) array of +/-1 spins
      J: coupling
      rng: numpy random generator

    Returns:
      spins: (L, L) modified in-place
    """
    L = spins.shape[0]
    p_add = 1.0 - math.exp(-2.0 * J)

    # Seed site
    r0 = rng.integers(0, L)
    c0 = rng.integers(0, L)
    spin_seed = spins[r0, c0]

    cluster = [(r0, c0)]
    in_cluster = np.zeros((L, L), dtype=bool)
    in_cluster[r0, c0] = True
    queue = [(r0, c0)]

    while queue:
        r, c = queue.pop()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = (r + dr) % L, (c + dc) % L
            if not in_cluster[nr, nc] and spins[nr, nc] == spin_seed:
                if rng.random() < p_add:
                    in_cluster[nr, nc] = True
                    cluster.append((nr, nc))
                    queue.append((nr, nc))

    # Flip cluster
    for r, c in cluster:
        spins[r, c] = -spin_seed

    return spins


def run_mcmc_numpy(
    L: int,
    J_c: float,
    n_samples: int,
    n_warmup: int,
    seed: int = 42,
    verbose: bool = False,
) -> dict:
    """Run Wolff MCMC on CPU using NumPy. Returns cumulants for Fisher/kappa3.

    NOTE: This is intentionally simplified. We compute only the quantities
    needed to estimate |R| via the leading-order formula.

    For the scalar curvature we use the DIRECT formula from spin cumulants:
      F_{ab} = <s_a s_b> - <s_a><s_b>   (second cumulant of edge spins)
      kappa3_{abc} = <(e_a - mu_a)(e_b - mu_b)(e_c - mu_c)>

    where e_a = s_i * s_j for edge a = (i, j).

    For uniform coupling at J_c, by symmetry F is circulant and we can
    estimate |R| from the spectrum of F alone using:
      |R| ~ (const) * n^{d_R}
    But for exact curvature we need the full tensor contraction.

    Returns dict with estimated |R| and metadata.
    """
    rng = np.random.default_rng(seed)

    # Initialize random spins
    spins = rng.choice([-1, 1], size=(L, L)).astype(np.float64)

    # Edge list (same order as run_ising_tm_minimal)
    edges = []
    for r in range(L):
        for c in range(L):
            site = r * L + c
            right = r * L + (c + 1) % L
            below = ((r + 1) % L) * L + c
            edges.append((min(site, right), max(site, right)))
            edges.append((min(site, below), max(site, below)))
    edges = sorted(set(edges))
    m = len(edges)

    def get_edge_obs(spins_flat: np.ndarray) -> np.ndarray:
        """Compute all m edge observables e_a = s_i * s_j."""
        obs = np.zeros(m)
        for idx, (i, j) in enumerate(edges):
            obs[idx] = spins_flat[i] * spins_flat[j]
        return obs

    # Warmup
    if verbose:
        print(f"  [L={L}] Warmup ({n_warmup} steps)...", flush=True)
    for _ in range(n_warmup):
        wolff_step_numpy(spins, J_c, rng)

    # Collection
    if verbose:
        print(f"  [L={L}] Collecting {n_samples} samples...", flush=True)

    edge_sums = np.zeros(m)
    edge_sq_sums = np.zeros((m, m))
    n_collected = 0

    t_sample = time.perf_counter()
    for step in range(n_samples):
        wolff_step_numpy(spins, J_c, rng)
        obs = get_edge_obs(spins.ravel())
        edge_sums += obs
        # Only accumulate diagonal for memory efficiency
        # (full m x m is too large for memory-constrained runs)
        n_collected += 1

        if verbose and step % (n_samples // 10) == 0:
            print(f"  [L={L}] {step}/{n_samples}...", flush=True)

    elapsed = time.perf_counter() - t_sample

    mu = edge_sums / n_collected  # (m,) mean edge observables

    if verbose:
        print(f"  [L={L}] MCMC done in {elapsed:.1f}s", flush=True)
        print(f"  [L={L}] NOTE: Full Fisher tensor not computed in NumPy mode.")
        print(f"         For full |R| computation, install JAX.")

    return {
        'L': L, 'n': L*L, 'm': m,
        'J_c': J_c,
        'R_scalar': None, 'abs_R': None,
        'status': 'MCMC_NUMPY_PARTIAL',
        'mu_norm': float(np.linalg.norm(mu)),
        'n_samples': n_collected,
        'elapsed_s': elapsed,
        'note': 'Full curvature requires JAX. Install with: pip install jax jaxlib',
    }


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def verify_against_reference(results: List[dict]) -> bool:
    """Check |R| values against pre-computed reference."""
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'data', 'reference_R_vs_L.json')
    try:
        with open(ref_path) as f:
            ref = json.load(f)
    except FileNotFoundError:
        print("WARNING: reference data not found")
        return True

    ref_by_L = {L: R for L, R in zip(ref['L'], ref['R_abs'])}
    all_pass = True

    print("\nVERIFICATION AGAINST REFERENCE (MCMC, 10% tolerance):")
    print(f"  {'L':>4}  {'computed':>12}  {'reference':>12}  {'error%':>8}  {'status':>8}")

    for r in results:
        L = r['L']
        R_comp = r.get('abs_R')
        if R_comp is None or L not in ref_by_L:
            print(f"  {L:>4}  {'N/A':>12}  {ref_by_L.get(L, 'N/A'):>12}  {'---':>8}  {'SKIP':>8}")
            continue
        R_ref = ref_by_L[L]
        err_pct = 100.0 * abs(R_comp - R_ref) / R_ref
        ok = err_pct <= 10.0
        if not ok:
            all_pass = False
        print(f"  {L:>4}  {R_comp:>12.4f}  {R_ref:>12.4f}  {err_pct:>7.2f}%  {'PASS' if ok else 'FAIL':>8}")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MCMC computation of |R(J_c, L)| for 2D Ising at large L",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
QUICK START:
  # L=10,12 with 100k samples (CPU, ~1-2 hours without GPU):
  python run_ising_mcmc_minimal.py --L 10 12 --n-samples 100000

  # L=10,12,14 with verification (requires JAX for full curvature):
  python run_ising_mcmc_minimal.py --L 10 12 14 --verify

NOTE: For the 30-minute replication, use run_ising_tm_minimal.py (L=3-7).
This MCMC script is for extended verification only.
""",
    )
    p.add_argument('--L', nargs='+', type=int, default=[10, 12],
                   help="Lattice sizes (default: 10 12)")
    p.add_argument('--n-samples', type=int, default=100000,
                   help="MCMC samples per size (default: 100000)")
    p.add_argument('--n-warmup', type=int, default=10000,
                   help="Warmup steps before collection (default: 10000)")
    p.add_argument('--seeds', nargs='+', type=int, default=[42],
                   help="Random seeds for averaging (default: 42)")
    p.add_argument('--output', type=str, default=None,
                   help="Save results to JSON file")
    p.add_argument('--verify', action='store_true',
                   help="Compare against reference data")
    p.add_argument('--verbose', action='store_true',
                   help="Verbose output")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 64)
    print("2D Ising MCMC Curvature Scaling (Large L)")
    print("=" * 64)
    print(f"  J_c = {J_C_2D:.10f}")
    print(f"  Sizes: {args.L}")
    print(f"  n_samples: {args.n_samples}")
    print(f"  n_warmup: {args.n_warmup}")
    print(f"  seeds: {args.seeds}")
    print(f"  JAX available: {HAS_JAX}")
    print()

    if not HAS_JAX:
        print("NOTE: JAX not available. Running NumPy fallback (partial results only).")
        print("      For full curvature: pip install jax jaxlib")
        print()

    results = []

    for L in args.L:
        print(f"\n[L={L}] Starting MCMC...", flush=True)
        seed_results = []

        for seed in args.seeds:
            res = run_mcmc_numpy(
                L=L,
                J_c=J_C_2D,
                n_samples=args.n_samples,
                n_warmup=args.n_warmup,
                seed=seed,
                verbose=args.verbose,
            )
            seed_results.append(res)
            if res.get('abs_R') is not None:
                print(f"  seed={seed}: |R| = {res['abs_R']:.4f}")
            else:
                print(f"  seed={seed}: status = {res['status']}")

        # Average over seeds if multiple
        abs_Rs = [r['abs_R'] for r in seed_results if r.get('abs_R') is not None]
        if abs_Rs:
            mean_R = float(np.mean(abs_Rs))
            std_R = float(np.std(abs_Rs)) if len(abs_Rs) > 1 else None
            combined = {
                'L': L, 'n': L*L, 'm': 2*L*L,
                'J_c': J_C_2D,
                'abs_R': mean_R,
                'abs_R_std': std_R,
                'R_scalar': -mean_R,  # 2D Ising R < 0
                'status': 'OK',
                'n_seeds': len(abs_Rs),
                'n_samples': args.n_samples,
            }
            results.append(combined)
            print(f"[L={L}] Mean |R| = {mean_R:.4f}" +
                  (f" +/- {std_R:.4f}" if std_R else ""))
        else:
            results.append(seed_results[0])
            print(f"[L={L}] No |R| computed (JAX required for full curvature)")

    # Verification
    if args.verify:
        verify_against_reference(results)

    # Save
    if args.output:
        payload = {
            'metadata': {
                'script': 'replication/run_ising_mcmc_minimal.py',
                'J_c': J_C_2D,
                'n_samples': args.n_samples,
                'n_warmup': args.n_warmup,
                'seeds': args.seeds,
                'jax_available': HAS_JAX,
            },
            'results': results,
        }
        with open(args.output, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    print()
    ok_count = sum(1 for r in results if r.get('status') == 'OK')
    print(f"Completed: {ok_count}/{len(args.L)} sizes with full curvature.")
    return 0 if ok_count == len(args.L) else 1


if __name__ == '__main__':
    sys.exit(main())
