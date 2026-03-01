# CLAIMS-STATUS.md — d_R = 10/9 Replication Package

What is proved, verified, or conjectured.

| # | Claim | Status | Evidence | Paper |
|---|-------|--------|----------|-------|
| 1 | Riem decomposition identity: R_lin = -2 R_quad for C_n | **Theorem (proved, C_n)** | Sage symbolic proof | Paper #7 §4 |
| 2 | Riem decomposition: R_lin = -2 R_quad for 42 graph families | **Numerically verified** | 839 graphs n<=7, machine precision | Paper #7 §4 |
| 3 | Tree => G_ab = 0 (vacuum Einstein condition, Ising h=0) | **Theorem (proved)** | Algebraic proof via tree decomposition | Paper #7 §5 |
| 4 | Non-tree => exists couplings with G_ab != 0 | **Conjecture + census** | 839 connected graphs n<=7 (100%) | Paper #7 §5 |
| 5 | d_R = 10/9 for 2D Ising (0.054% from 10/9) | **Numerically verified** | Exact TM L=3-9 + MCMC L=10-20 | Papers #8, PL |
| 6 | d_R formula: (d nu + 2 eta)/(d nu + eta) from CFT | **Derived + verified** | 2D Ising, 3D Ising, Potts q=3 | Prediction Letter |
| 7 | Scalar Ricci identity (Riem tensor identity, Ising h=0) | **Proved (C_n), verified (42 families)** | Sage + numeric | Paper #7 §4 |
| 8 | d_R = 10/9 for 2D Ising (analytic proof, Gap A) | **Open (numerically closed)** | Richardson extrapolation to 0.054% | Paper #8 §5.1 |
| 9 | d_R formula universality for all CFTs | **Conjecture** | Verified: 2D Ising, 3D Ising, Potts q=3,4 | Prediction Letter |
| 10 | Correction-to-scaling: ω = d_R (self-referential) | **Empirical conjecture** | AICc: fixed ω=10/9 wins over free ω by +17 | Paper #8 |

## Key Numbers for Replication

| Quantity | Value | Source |
|----------|-------|--------|
| J_c (2D Ising) | 0.44068679350977147 | Onsager exact: log(1+sqrt(2))/2 |
| d_R (2D Ising) | 10/9 = 1.1111... | CFT: (d nu + 2 eta)/(d nu + eta) |
| d_R measured | 1.111 ± 0.002 | MCMC L=10-20 consecutive d_eff mean |
| Deviation from 10/9 | 0.054% | Richardson extrapolation |
| |R(J_c, L=3)| | 105.64 | Exact TM |
| |R(J_c, L=10)| | 2091.9 | MCMC mean (3 seeds, 500k samples) |
| |R(J_c, L=20)| | 9785.3 | MCMC mean (3 seeds, 100k samples) |

## CFT Exponents (2D Ising)

| Exponent | Symbol | Value |
|----------|--------|-------|
| Correlation length | nu | 1 |
| Anomalous dim | eta | 1/4 |
| Dimension | d | 2 |
| d_R formula | (d nu + 2 eta)/(d nu + eta) | 2.5/2.25 = **10/9** |

## Evidence Hierarchy

1. **Proved**: Algebraic theorem with complete proof in paper.
2. **Numerically verified**: Computed to high precision, consistent with claim.
3. **Conjecture + census**: Verified on all small cases, no counterexample found.
4. **Open**: Claim follows numerically but analytic proof is missing.

## Gap A Status

Gap A is the missing analytic proof of the d_R = 10/9 scaling exponent from
first principles (Pfaffian/Szego asymptotics of the transfer matrix spectrum).

**Current status**: Numerically verified to 0.054% across L=3-20. The analytic
proof would require establishing the asymptotic form of the Fisher information
trace via the Kasteleyn-Berezin-Temperley free-fermion representation.

This is an open research problem. The numerical evidence is strong (13 data
points, consistent CFT prediction, 4 independent evidence streams), but the
mathematical proof gap remains.

## What an External Replicator Can Verify in 30 Minutes

1. Run `run_ising_tm_minimal.py --L 3 4 5 6` (exact, ~5 min).
2. Check |R| values match reference data within 2%.
3. Observe d_eff > 1.1 and decreasing toward 10/9.
4. Run `analyze_scaling.py` to see the convergence plot.

The MCMC data for L=10-20 requires JAX and GPU (or many hours on CPU) to
recompute from scratch. The reference JSON provides pre-computed values.
