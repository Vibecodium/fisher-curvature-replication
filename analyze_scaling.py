"""
analyze_scaling.py
==================
Load |R(J_c, L)| data (reference or freshly computed) and verify d_R = 10/9.

Produces:
  - Console report: d_R fit, d_eff convergence, deviation from 10/9
  - Publication-quality figure: log|R| vs log(n) with fit line

USAGE:
  # Analyze reference data (no computation needed):
  python analyze_scaling.py

  # Analyze freshly computed results:
  python analyze_scaling.py --input results.json

  # Save figure:
  python analyze_scaling.py --figure figures/dR_scaling.pdf

  # Combine reference + computed data:
  python analyze_scaling.py --input results.json --include-reference

WHAT THIS VERIFIES:
  The claim: |R(J_c, L)| ~ A * n^{d_R}  as L -> infinity
  where n = L^2 (number of sites) and d_R = 10/9 = 1.111...

  The CFT prediction:
    d_R = (d * nu + 2 * eta) / (d * nu + eta)
  For 2D Ising: d=2, nu=1, eta=1/4
    d_R = (2*1 + 2*0.25) / (2*1 + 0.25) = 2.5 / 2.25 = 10/9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

REFERENCE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'reference_R_vs_L.json',
)


def load_reference() -> Tuple[List[int], List[float], List[str]]:
    """Load reference |R| data from data/reference_R_vs_L.json.

    Returns:
      Ls: list of L values
      Rs: list of |R| values
      sources: list of source descriptions per data point
    """
    with open(REFERENCE_PATH) as f:
        ref = json.load(f)
    Ls = ref['L']
    Rs = ref['R_abs']
    src = ref['source']
    sources = []
    for L in Ls:
        if L <= 9:
            sources.append(src.get('L_3_to_9', 'exact TM'))
        else:
            sources.append(src.get('L_10_to_20', 'MCMC'))
    return Ls, Rs, sources


def load_computed(path: str) -> Tuple[List[int], List[float]]:
    """Load freshly computed |R| data from JSON output of run_ising_tm_minimal.py.

    Returns:
      Ls: list of L values
      Rs: list of |R| values (None for failed computations)
    """
    with open(path) as f:
        data = json.load(f)

    results = data.get('results', [])
    Ls, Rs = [], []
    for r in results:
        if r.get('status') == 'OK' and r.get('abs_R') is not None:
            Ls.append(int(r['L']))
            Rs.append(float(r['abs_R']))
    return Ls, Rs


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def fit_power_law(
    Ls: List[int],
    Rs: List[float],
    min_L: int = 1,
) -> dict:
    """Fit |R| ~ A * n^{d_R} in log-log space where n = L^2.

    Args:
      Ls: lattice sizes
      Rs: |R| values
      min_L: exclude L < min_L from fit (to reduce finite-size contamination)

    Returns:
      dict with d_R, A, R_squared, n_pts, std_err_d_R
    """
    pairs = [(L*L, R) for L, R in zip(Ls, Rs)
             if R is not None and R > 0 and L >= min_L]
    if len(pairs) < 2:
        return {'d_R': None, 'A': None, 'R_squared': None, 'n_pts': len(pairs)}

    log_n = np.array([math.log(n) for n, _ in pairs])
    log_R = np.array([math.log(R) for _, R in pairs])

    mx = log_n.mean()
    my = log_R.mean()
    ssxy = ((log_n - mx) * (log_R - my)).sum()
    ssxx = ((log_n - mx)**2).sum()

    if ssxx < 1e-14:
        return {'d_R': None, 'A': None, 'R_squared': None, 'n_pts': len(pairs)}

    slope = ssxy / ssxx
    intercept = my - slope * mx
    R_pred = intercept + slope * log_n
    ss_res = ((log_R - R_pred)**2).sum()
    ss_tot = ((log_R - my)**2).sum()
    R_sq = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0

    # Standard error of slope
    n_pts = len(pairs)
    if n_pts > 2 and ss_res > 0:
        s2 = ss_res / (n_pts - 2)
        std_err = math.sqrt(s2 / ssxx)
    else:
        std_err = None

    return {
        'd_R': float(slope),
        'A': float(math.exp(intercept)),
        'R_squared': float(R_sq),
        'n_pts': n_pts,
        'std_err_d_R': float(std_err) if std_err is not None else None,
    }


def compute_d_eff_sequence(Ls: List[int], Rs: List[float]) -> List[dict]:
    """Compute d_eff from consecutive (L, |R|) pairs.

    d_eff(n->n') = log(|R(n')|/|R(n)|) / log(n'/n)  where n = L^2.

    This sequence should converge to d_R = 10/9 from above as L increases.
    Convergence rate is governed by the leading correction-to-scaling exponent.
    """
    result = []
    for i in range(len(Ls) - 1):
        L1, R1 = Ls[i], Rs[i]
        L2, R2 = Ls[i+1], Rs[i+1]
        if R1 is None or R2 is None or R1 <= 0 or R2 <= 0:
            continue
        n1, n2 = L1*L1, L2*L2
        d_eff = math.log(R2/R1) / math.log(n2/n1)
        result.append({
            'L_from': L1, 'L_to': L2,
            'n_from': n1, 'n_to': n2,
            'd_eff': d_eff,
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# PRINTING
# ─────────────────────────────────────────────────────────────────────────────

def print_report(
    Ls: List[int],
    Rs: List[float],
    fit: dict,
    d_effs: List[dict],
    label: str = 'data',
) -> None:
    """Print a formatted analysis report to stdout."""
    D_R_PREDICTED = 10.0 / 9.0

    print()
    print("=" * 64)
    print(f"SCALING ANALYSIS: d_R = 10/9 Verification ({label})")
    print("=" * 64)

    print()
    print(f"  {'L':>4}  {'n=L^2':>7}  {'|R(J_c,L)|':>12}  {'log|R|':>8}")
    print(f"  {'-'*4}  {'-'*7}  {'-'*12}  {'-'*8}")
    for L, R in zip(Ls, Rs):
        if R is not None and R > 0:
            print(f"  {L:>4}  {L*L:>7}  {R:>12.4f}  {math.log(R):>8.4f}")

    print()
    print(f"  POWER LAW FIT: |R| ~ A * n^{{d_R}}")
    if fit['d_R'] is not None:
        d_R = fit['d_R']
        dev = d_R - D_R_PREDICTED
        pct = 100.0 * dev / D_R_PREDICTED
        print(f"    d_R = {d_R:.4f}")
        if fit.get('std_err_d_R') is not None:
            print(f"    std_err(d_R) = {fit['std_err_d_R']:.4f}")
        print(f"    A   = {fit['A']:.4f}")
        print(f"    R^2 = {fit['R_squared']:.6f}  (n_pts = {fit['n_pts']})")
        print()
        print(f"    Predicted: d_R = 10/9 = {D_R_PREDICTED:.6f}")
        print(f"    Deviation: {dev:+.4f} ({pct:+.2f}%)")
        print()
        if abs(pct) < 5.0:
            print("    STATUS: CONSISTENT with d_R = 10/9")
        else:
            print(f"    STATUS: {abs(pct):.1f}% deviation -- may need larger L for convergence")
    else:
        print("    INSUFFICIENT DATA for fit")

    print()
    print("  CONSECUTIVE d_eff (converging to 10/9 = 1.1111 from above):")
    print(f"    {'L1->L2':>9}  {'d_eff':>8}  {'delta':>9}")
    if d_effs:
        for de in d_effs:
            delta = de['d_eff'] - D_R_PREDICTED
            print(f"    {de['L_from']}->{de['L_to']:>4}    {de['d_eff']:>8.4f}  {delta:>+8.4f}")
        last_d_eff = d_effs[-1]['d_eff']
        print()
        print(f"    Last d_eff: {last_d_eff:.4f} (target: {D_R_PREDICTED:.4f})")
        gap = last_d_eff - D_R_PREDICTED
        if gap > 0:
            print(f"    Still {gap:.4f} above target -- converging from above (expected)")
    else:
        print("    (Need >= 2 data points)")

    print()
    print("  CFT FORMULA:")
    print("    d_R = (d*nu + 2*eta) / (d*nu + eta)")
    print("    2D Ising: d=2, nu=1, eta=1/4")
    print(f"    d_R = (2 + 0.5) / (2 + 0.25) = 2.5 / 2.25 = 10/9 = {D_R_PREDICTED:.6f}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def make_figure(
    Ls_ref: List[int],
    Rs_ref: List[float],
    Ls_comp: Optional[List[int]] = None,
    Rs_comp: Optional[List[float]] = None,
    fit: Optional[dict] = None,
    output_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Generate publication-quality log|R| vs log(n) figure.

    Args:
      Ls_ref:    reference L values (exact TM + MCMC)
      Rs_ref:    reference |R| values
      Ls_comp:   freshly computed L values (optional overlay)
      Rs_comp:   freshly computed |R| values (optional overlay)
      fit:       power law fit dict (from fit_power_law)
      output_path: save figure to this path (PDF or PNG)
      show:      display interactive figure
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib not installed -- skipping figure generation.")
        print("Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Left panel: log|R| vs log(n) with fit line ──
    ax = axes[0]
    D_R_PREDICTED = 10.0 / 9.0

    # Reference data: split by source
    Ls_tm = [L for L in Ls_ref if L <= 9]
    Rs_tm = [R for L, R in zip(Ls_ref, Rs_ref) if L <= 9]
    Ls_mcmc = [L for L in Ls_ref if L > 9]
    Rs_mcmc = [R for L, R in zip(Ls_ref, Rs_ref) if L > 9]

    ns_tm = [L*L for L in Ls_tm]
    ns_mcmc = [L*L for L in Ls_mcmc]

    if ns_tm and Rs_tm:
        ax.scatter(np.log(ns_tm), np.log(Rs_tm),
                   marker='o', s=60, color='royalblue', zorder=5,
                   label='Exact TM (L=3-9)')
    if ns_mcmc and Rs_mcmc:
        ax.scatter(np.log(ns_mcmc), np.log(Rs_mcmc),
                   marker='s', s=60, color='darkorange', zorder=5,
                   label='MCMC (L=10-20)')

    # Freshly computed overlay
    if Ls_comp and Rs_comp:
        ns_comp = [L*L for L in Ls_comp]
        ax.scatter(np.log(ns_comp), np.log(Rs_comp),
                   marker='D', s=80, color='green', zorder=6,
                   facecolors='none', linewidths=2,
                   label='Computed (this run)')

    # Fit line
    if fit and fit.get('d_R') is not None:
        all_ns = [L*L for L in Ls_ref]
        log_n_min = math.log(min(all_ns)) - 0.1
        log_n_max = math.log(max(all_ns)) + 0.1
        n_line = np.linspace(log_n_min, log_n_max, 100)
        log_A = math.log(fit['A'])
        ax.plot(n_line, log_A + fit['d_R'] * n_line,
                'k--', linewidth=1.5,
                label=f"Fit: $d_R = {fit['d_R']:.4f}$")

    # Predicted slope line (anchored at L=10 reference)
    if Ls_ref and Rs_ref:
        anchor_L = 10
        if anchor_L in Ls_ref:
            idx_anchor = Ls_ref.index(anchor_L)
            log_R_anchor = math.log(Rs_ref[idx_anchor])
            log_n_anchor = math.log(anchor_L**2)
            log_n_min = math.log(min(L*L for L in Ls_ref)) - 0.2
            log_n_max = math.log(max(L*L for L in Ls_ref)) + 0.2
            n_line = np.linspace(log_n_min, log_n_max, 100)
            ax.plot(n_line,
                    log_R_anchor + D_R_PREDICTED * (n_line - log_n_anchor),
                    'r-', linewidth=2, alpha=0.7,
                    label=f"Predicted: $d_R = 10/9 = {D_R_PREDICTED:.4f}$")

    ax.set_xlabel(r'$\log(n)$  where $n = L^2$', fontsize=13)
    ax.set_ylabel(r'$\log |R(J_c, L)|$', fontsize=13)
    ax.set_title(r'Fisher Curvature Scaling: 2D Ising', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # ── Right panel: d_eff convergence ──
    ax2 = axes[1]
    all_Ls = Ls_ref[:]
    all_Rs = Rs_ref[:]
    if Ls_comp and Rs_comp:
        # Merge, prefer computed values
        for L, R in zip(Ls_comp, Rs_comp):
            if L not in all_Ls:
                all_Ls.append(L)
                all_Rs.append(R)
        # Sort by L
        pairs = sorted(zip(all_Ls, all_Rs))
        all_Ls = [p[0] for p in pairs]
        all_Rs = [p[1] for p in pairs]

    d_effs = compute_d_eff_sequence(all_Ls, all_Rs)
    if d_effs:
        L_mids = [(de['L_from'] + de['L_to']) / 2.0 for de in d_effs]
        d_eff_vals = [de['d_eff'] for de in d_effs]
        ax2.plot(L_mids, d_eff_vals, 'bo-', markersize=6, linewidth=1.5,
                 label=r'$d_{\rm eff}(L \to L+\delta)$')
        ax2.axhline(D_R_PREDICTED, color='red', linestyle='--', linewidth=2,
                    label=f'$d_R = 10/9 = {D_R_PREDICTED:.4f}$ (predicted)')
        ax2.set_xlim(left=0)
        margin = 0.05
        y_min = min(D_R_PREDICTED - margin, min(d_eff_vals) - margin)
        y_max = max(d_eff_vals) + margin
        ax2.set_ylim(y_min, y_max)

    ax2.set_xlabel(r'$L$ (approximate)', fontsize=13)
    ax2.set_ylabel(r'$d_{\rm eff}$', fontsize=13)
    ax2.set_title(r'Consecutive $d_{\rm eff}$ Convergence', fontsize=14)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Analyze |R| scaling data and verify d_R = 10/9",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze reference data only:
  python analyze_scaling.py

  # Analyze freshly computed results:
  python analyze_scaling.py --input results.json

  # Save figure:
  python analyze_scaling.py --figure figures/dR_scaling.pdf

  # No interactive plot:
  python analyze_scaling.py --no-show --figure figures/dR_scaling.png
""",
    )
    p.add_argument('--input', type=str, default=None,
                   help="JSON file from run_ising_tm_minimal.py (optional)")
    p.add_argument('--include-reference', action='store_true',
                   help="Also include reference data when --input is given")
    p.add_argument('--figure', type=str, default=None,
                   help="Save figure to this path (PDF or PNG)")
    p.add_argument('--no-show', action='store_true',
                   help="Do not display interactive figure")
    p.add_argument('--min-L-fit', type=int, default=5,
                   help="Minimum L for power law fit (default: 5, reduces finite-size bias)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    Ls_comp, Rs_comp = None, None

    if args.input:
        print(f"Loading computed data from: {args.input}")
        Ls_comp, Rs_comp = load_computed(args.input)
        print(f"  Found {len(Ls_comp)} successful computations: L = {Ls_comp}")

    Ls_ref, Rs_ref, sources = load_reference()
    print(f"Loaded reference data: L = {Ls_ref}")

    # Decide which data to analyze
    if args.input and not args.include_reference:
        # Analyze only freshly computed
        Ls_analysis = Ls_comp
        Rs_analysis = Rs_comp
        label = f"computed ({os.path.basename(args.input)})"
    elif args.input and args.include_reference:
        # Merge: use computed where available, reference elsewhere
        by_L = {L: R for L, R in zip(Ls_ref, Rs_ref)}
        for L, R in zip(Ls_comp, Rs_comp):
            by_L[L] = R  # computed overrides reference
        Ls_analysis = sorted(by_L.keys())
        Rs_analysis = [by_L[L] for L in Ls_analysis]
        label = "computed + reference"
    else:
        # Reference only
        Ls_analysis = Ls_ref
        Rs_analysis = Rs_ref
        label = "reference data"

    # Fit
    fit = fit_power_law(Ls_analysis, Rs_analysis, min_L=args.min_L_fit)
    d_effs = compute_d_eff_sequence(Ls_analysis, Rs_analysis)

    # Report
    print_report(Ls_analysis, Rs_analysis, fit, d_effs, label=label)

    # Figure
    if args.figure or not args.no_show:
        make_figure(
            Ls_ref=Ls_ref,
            Rs_ref=Rs_ref,
            Ls_comp=Ls_comp,
            Rs_comp=Rs_comp,
            fit=fit,
            output_path=args.figure,
            show=not args.no_show,
        )

    print()
    print("Analysis complete.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
