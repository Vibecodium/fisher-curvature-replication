"""
test_minimal.py
===============
One-command validation of the d_R = 10/9 replication package.

Computes |R(J_c, L=3)| using the exact transfer matrix and checks it matches
the reference value 105.64 within 1%.

Usage:
    python test_minimal.py

Exit code:
    0 if PASS
    1 if FAIL
"""

import sys

# Reference value for L=3, exact transfer matrix.
# Source: data/reference_R_vs_L.json (exact TM, reproducible).
REFERENCE_ABS_R_L3 = 105.64
TOLERANCE_FRACTION = 0.01  # 1% tolerance


def main() -> int:
    """Run the minimal validation test.

    Returns:
        0 on PASS, 1 on FAIL.
    """
    print("=" * 60)
    print("Minimal replication test: d_R = 10/9")
    print("Computing |R(J_c, L=3)| via exact transfer matrix...")
    print("=" * 60)

    # Import from run_ising_tm_minimal (same directory).
    # We import here (not at module top) so import errors produce a clear message.
    try:
        from run_ising_tm_minimal import compute_R_for_L
    except ImportError as exc:
        print(f"FAIL: cannot import run_ising_tm_minimal: {exc}")
        print("Make sure you are running from inside the replication/ directory")
        print("and have installed requirements: pip install -r requirements.txt")
        return 1

    # Run exact TM for L=3 (should finish in < 1 second).
    result = compute_R_for_L(L=3)

    if result.get("status") != "OK":
        print(f"FAIL: compute_R_for_L returned status={result.get('status')!r}")
        return 1

    abs_R = result["abs_R"]
    if abs_R is None:
        print("FAIL: abs_R is None (singular Fisher matrix?)")
        return 1

    error_fraction = abs(abs_R - REFERENCE_ABS_R_L3) / REFERENCE_ABS_R_L3

    print()
    print(f"  L           = 3")
    print(f"  J_c         = {result['J_c']:.14f}")
    print(f"  |R| computed = {abs_R:.4f}")
    print(f"  |R| reference= {REFERENCE_ABS_R_L3:.4f}")
    print(f"  error        = {100.0 * error_fraction:.4f}%")
    print(f"  tolerance    = {100.0 * TOLERANCE_FRACTION:.1f}%")
    print()

    if error_fraction <= TOLERANCE_FRACTION:
        print("PASS")
        return 0
    else:
        print(
            f"FAIL: computed |R|={abs_R:.4f} deviates from reference "
            f"{REFERENCE_ABS_R_L3:.4f} by {100.0 * error_fraction:.2f}% "
            f"(tolerance {100.0 * TOLERANCE_FRACTION:.1f}%)"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
