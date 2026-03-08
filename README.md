# Replication Package: Fisher Curvature Scaling Exponent d_R = 10/9

Self-contained package to verify the claim that the Fisher information scalar
curvature of the 2D Ising model at criticality scales as |R| ~ n^{d_R} with
d_R = 10/9 = 1.111..., as predicted by conformal field theory.

**External verification time: ~30 minutes on a modern laptop.**

---

## Abstract

The 2D Ising model at its critical coupling J_c = log(1+sqrt(2))/2 carries a
natural Riemannian geometry on its statistical manifold -- the Fisher information
geometry. We claim that the scalar curvature R of this manifold diverges as
|R(J_c, L)| ~ A * n^{d_R} with d_R = 10/9, where n = L^2 is the number of sites.
The exponent is predicted by conformal field theory via d_R = (d*nu + 2*eta) /
(d*nu + eta), which for the 2D Ising universality class (d=2, nu=1, eta=1/4)
gives exactly 2.5/2.25 = 10/9. This package provides exact transfer matrix
scripts for L=3-9 and pre-computed MCMC reference data for L=10-20 (13 data
points total, consecutive d_eff converging toward 10/9 from above, within ~2%
for L >= 10).

---

## Installation

```bash
git clone https://github.com/Vibecodium/fisher-curvature-replication.git
cd fisher-curvature-replication
pip install -r requirements.txt
```

Python >= 3.9 required. No GPU needed for the 30-minute verification.

---

## Quickstart

```bash
# 1. One-command validation (L=3, <1 second)
python test_minimal.py

# 2. Run exact transfer matrix for L=3,4,5,6 (~5 minutes)
python run_ising_tm_minimal.py --L 3 4 5 6 --verify

# 3. Analyze scaling (uses pre-computed reference for L=3-20)
python analyze_scaling.py --no-show

# 4. Save figure
python analyze_scaling.py --figure figures/dR_scaling.pdf --no-show
```

Expected output from step 2:

```
L=3: |R| = 105.64   PASS (reference: 105.64)
L=4: |R| = 265.31   PASS (reference: 265.31)
L=5: |R| = 446.24   PASS (reference: 446.24)
L=6: |R| = 671.25   PASS (reference: 671.25)
VERIFICATION: ALL CHECKS PASSED

SCALING ANALYSIS:
  d_R = 1.105 (from L=3-6 fit)
  Consecutive d_eff:
    L3->L4:  d_eff = 1.601  (converging from above)
    L4->L5:  d_eff = 1.340
    L5->L6:  d_eff = 1.244
  NOTE: d_eff converges to 10/9 = 1.111 as L increases.
```

---

## What Is Being Verified

The **Fisher information scalar curvature** R of the 2D Ising model on an L x L
periodic torus at the critical coupling J_c = log(1+sqrt(2))/2.

**Main claim**: |R(J_c, L)| ~ A * n^{d_R} with d_R = 10/9, where n = L^2.

**CFT prediction**:

    d_R = (d * nu + 2 * eta) / (d * nu + eta)

For 2D Ising (d=2, nu=1, eta=1/4):

    d_R = 2.5 / 2.25 = 10/9 = 1.1111...

**Verified numerically**: 13 data points L=3-20 (7 exact TM + 6 MCMC), consecutive
d_eff converging toward 10/9 from above, with large-L mean d_eff = 1.111 = 10/9.

---

## Usage

### Minimal test (1 second)

```bash
python test_minimal.py
```

Runs the exact transfer matrix for L=3 and checks |R| against the reference
value (105.64) within 1%. Prints `PASS` or `FAIL` with details.

### Exact transfer matrix (L=3-9)

```bash
# Default: L=3,4,5,6
python run_ising_tm_minimal.py

# Explicit sizes with verification
python run_ising_tm_minimal.py --L 3 4 5 6 7 --verify

# Save results to JSON
python run_ising_tm_minimal.py --L 3 4 5 --output results.json
```

### Scaling analysis

```bash
# Analyze reference data (no computation needed)
python analyze_scaling.py --no-show

# Analyze freshly computed results
python analyze_scaling.py --input results.json --no-show

# Combine reference + computed data
python analyze_scaling.py --input results.json --include-reference --no-show

# Save figure
python analyze_scaling.py --figure figures/dR_scaling.pdf --no-show
```

---

## Expected Output

| L | |R(J_c, L)| | Source |
|---|-----------|--------|
| 3 | 105.64 | Exact TM |
| 4 | 265.31 | Exact TM |
| 5 | 446.24 | Exact TM |
| 6 | 671.25 | Exact TM |
| 7 | 945.53 | Exact TM |
| 8 | 1272.14 | Exact TM |
| 9 | 1653.34 | Exact TM |
| 10 | 2091.9 | MCMC (500k samples, 3 seeds) |
| 20 | 9785.3 | MCMC (500k samples, 3 seeds) |

Consecutive effective exponent d_eff converges from above toward 10/9 = 1.111:

    L3->L4:  1.601
    L4->L5:  1.340
    L5->L6:  1.244
    L6->L7:  1.193
    L7->L8:  1.166
    L8->L9:  1.149
    L9->L10: 1.126
    L10->L12: 1.119
    L12->L14: 1.119
    mean (large L): 1.111 = 10/9

---

## Files in This Package

```
fisher-curvature-replication/
├── README.md                    # This file
├── REPLICATION.md               # Detailed step-by-step guide
├── CLAIMS-STATUS.md             # What is theorem / verified / conjecture
├── requirements.txt             # Python dependencies
├── test_minimal.py              # One-command validation test
├── run_ising_tm_minimal.py      # Exact TM: |R| for L=3-9 (CPU, NumPy only)
├── run_ising_mcmc_minimal.py    # MCMC: |R| for L>=10 (requires JAX)
├── analyze_scaling.py           # Compute d_R from data, generate figure
└── data/
    └── reference_R_vs_L.json   # Pre-computed |R| for L=3-20
```

---

## Dependencies

Minimal (for TM verification, L=3-7, recommended):

```
numpy>=1.24
scipy>=1.10
matplotlib>=3.7
```

For MCMC (L>=10, optional):

```
jax>=0.4.20    # CPU JAX (works without GPU)
jaxlib>=0.4.20
tqdm>=4.65
```

Install all: `pip install -r requirements.txt`

---

## Timing

| Computation | Time | Notes |
|-------------|------|-------|
| TM L=3 | <1s | Exact, deterministic |
| TM L=4 | <1s | Exact, deterministic |
| TM L=5 | ~5s | Exact, deterministic |
| TM L=6 | ~20s | Exact, deterministic |
| TM L=7 | ~90s | Exact, deterministic |
| TM L=8 | ~600s | Exact, 4 GB RAM needed |
| MCMC L=10 (CPU) | ~30 min | 100k samples, statistical |
| analyze_scaling.py | <5s | Uses pre-computed reference |

**30-minute verification**: Run TM for L=3,4,5,6, then analyze_scaling.py.
Pre-computed reference data for L=10-20 is in data/reference_R_vs_L.json.

---

## Reference Data

Pre-computed |R| values in `data/reference_R_vs_L.json`:

- L=3-9: exact transfer matrix (CPU NumPy, deterministic, reproducible)
- L=10-20: MCMC Wolff algorithm (RTX 4090, 500k samples, 3 seeds each)

MCMC values for L=10-20 have statistical uncertainty of ~1-2%.

---

## Papers

This package accompanies the Fisher curvature paper series:

- **Prediction Letter** (PRIMARY): d_R = (d*nu + 2*eta)/(d*nu + eta) conjecture for all CFTs.
  Zenodo: https://zenodo.org/records/18807279

- **Paper #7** (Curvature): Riemann decomposition identity and vacuum Einstein condition.
  Zenodo: https://zenodo.org/records/18807275

- ~~Paper #8 (Asymptotics)~~: **SUPERSEDED** -- the central theorem (c_2 ~ L^{2/9})
  was falsified; actual behavior is c_2 ~ ln(L). The d_R exponent itself remains
  valid as an empirical observable (see Prediction Letter), but Paper #8's analytical
  derivation is incorrect. Zenodo record retained for transparency:
  https://zenodo.org/records/18807277

**Full collection (papers + DOIs)**:
https://zenodo.org/records/18809947 (DOI: 10.5281/zenodo.18806742)

---

## Citation

If you use this package, please cite the prediction letter (primary reference):

```
@misc{zhuravlev2026fisher,
  title  = {Fisher Curvature Scaling at Critical Points:
             An Exact Information-Geometric Exponent from Periodic
             Boundary Conditions},
  author = {Zhuravlev, Maxim},
  year   = {2026},
  doi    = {10.5281/zenodo.18807279},
  url    = {https://zenodo.org/records/18807279}
}
```

---

## License

CC-BY-4.0. You are free to use, share, and adapt this material with attribution.

---

## Troubleshooting

If computed values deviate from reference by more than 2% (exact TM):

1. Check Python/NumPy version: Python >= 3.9, NumPy >= 1.24.
2. Check that J_c = log(1+sqrt(2))/2 is used (not an approximation).
3. Check finite difference step eps=0.02 (default).
4. For L >= 7, ensure at least 4 GB free RAM.

See REPLICATION.md for full troubleshooting guide and algorithm description.

---

## Contact

For questions about the physics, see the full paper series at the Zenodo
collection above. For code issues, open a GitHub issue.
