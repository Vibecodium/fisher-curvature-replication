# CLAIMS-STATUS.md — d_R = 10/9 Replication Package

What is proved, verified, or conjectured.

| # | Claim | Status | Evidence | Paper |
|---|-------|--------|----------|-------|
| 1 | Riem decomposition identity: R_lin = -2 R_quad for C_n | **Theorem (proved, C_n)** | Sage symbolic proof | Paper #7 §4 |
| 2 | Riem decomposition: R_lin = -2 R_quad for 42 graph families | **Numerically verified** | 839 graphs n<=7, machine precision | Paper #7 §4 |
| 3 | Tree => G_ab = 0 (vacuum Einstein condition, Ising h=0) | **Theorem (proved)** | Algebraic proof via tree decomposition | Paper #7 §5 |
| 4 | Non-tree => exists couplings with G_ab != 0 | **Conjecture + census** | 839 connected graphs n<=7 (100%) | Paper #7 §5 |
| 5 | d_R = 10/9 for 2D Ising (large-L d_eff mean = 1.111) | **Numerically verified** | Exact TM L=3-9 + MCMC L=10-20 | Prediction Letter |
| 6 | d_R formula: (d nu + 2 eta)/(d nu + eta) from CFT | **Derived + verified** | 2D Ising, 3D Ising, Potts q=3 | Prediction Letter |
| 7 | Scalar Ricci identity (Riem tensor identity, Ising h=0) | **Proved (C_n), verified (42 families)** | Sage + numeric | Paper #7 §4 |
| 8 | d_R = 10/9 for 2D Ising (analytic proof, Gap A) | **Open** | Paper #8 approach falsified (c_2 ~ ln(L), not L^{2/9}); numerical evidence from |R| scaling remains strong | -- |
| 9 | d_R formula universality for all CFTs | **Conjecture** | Verified: 2D Ising, 3D Ising, Potts q=3,4 | Prediction Letter |
| 10 | Correction-to-scaling: omega = d_R (self-referential) | **Empirical conjecture** | AICc: fixed omega=10/9 wins over free omega by +17 | ~~Paper #8~~ (falsified; empirical observation may still hold independently) |

## Key Numbers for Replication

| Quantity | Value | Source |
|----------|-------|--------|
| J_c (2D Ising) | 0.44068679350977147 | Onsager exact: log(1+sqrt(2))/2 |
| d_R (2D Ising) | 10/9 = 1.1111... | CFT: (d nu + 2 eta)/(d nu + eta) |
| d_R measured | 1.111 ± 0.002 | MCMC L=10-20 consecutive d_eff mean |
| Large-L d_eff mean | 1.111 = 10/9 | Consecutive d_eff from |R| scaling, L=10-20 |
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
first principles.

**Current status**: The approach via Paper #8 (c_2 ~ L^{2/9} asymptotics) has
been **falsified** -- ultra-fine transfer-matrix data (L=4-800) shows c_2 grows
sub-logarithmically as c_2 ~ ln(L), not as a power law. The d_R = 10/9 exponent
itself remains numerically verified (13 data points, large-L d_eff mean = 10/9),
but the analytical derivation route through single-mode c_2 asymptotics is
invalid. The exponent emerges as a collective property of the full Brillouin zone
sum, not from any single Fourier mode.

This is an open research problem. The numerical evidence is strong (13+ data
points, consistent CFT prediction across multiple universality classes), but the
mathematical proof gap remains and requires a new analytical approach.

## What an External Replicator Can Verify in 30 Minutes

1. Run `run_ising_tm_minimal.py --L 3 4 5 6` (exact, ~5 min).
2. Check |R| values match reference data within 2%.
3. Observe d_eff > 1.1 and decreasing toward 10/9.
4. Run `analyze_scaling.py` to see the convergence plot.

The MCMC data for L=10-20 requires JAX and GPU (or many hours on CPU) to
recompute from scratch. The reference JSON provides pre-computed values.
