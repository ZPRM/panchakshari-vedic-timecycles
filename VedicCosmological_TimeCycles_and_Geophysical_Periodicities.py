#!/usr/bin/env python3
"""
VedicCosmological_TimeCycles_and_Geophysical_Periodicities.py
===============================================================
Monte Carlo permutation test: Does the Vedic multiplier *architecture*
outperform size-matched random architectures in matching geophysical
periodicities?

Author : Pramod Chilakalapudi
Notebook: Panchakshari_Paper_CSALT_Verify_v2.ipynb (C-SALT verification)
Project: P1 – Pañcākṣarī Pravachanam
Paper  : Vedic Cosmological Time Cycles and Geophysical Periodicities (v8.6)
Date   : 2026-03-14

Usage
-----
    python vedic_mc_simulation.py            # runs all tests, prints report
    python vedic_mc_simulation.py --seed 42  # reproducible run
    python vedic_mc_simulation.py --n 100000 # higher precision

Null Hypothesis
---------------
A randomly-chosen set of K multipliers, drawn from the same log-uniform
range as the Vedic set and applied to the same base number, achieves
≥ M geophysical overlaps by chance.

Key design choices (fully specified to ensure reproducibility):
  - Base number   : 432,000 yr  (held fixed in all runs)
  - Overlap criterion: |V - G| / sigma_G  ≤ 1.0   (1-sigma)
  - Null distribution: log-uniform over [min_mult, max_mult]
    where min_mult / max_mult are the min/max of the Vedic multiplier set
    (so the random sets span *exactly* the same multiplicative range)
  - K (number of multipliers): same as Vedic set (7)
  - N simulations: 100,000 by default
"""

import argparse
import numpy as np

# ── Geophysical comparators ──────────────────────────────────────────────────
# Each entry: (geo_value_yr, sigma_yr, label, source)
# sigma_yr = 1-sigma uncertainty from primary literature.
GEO = [
    (4.32e5,  2.5e4,  "Geomagnetic reversal mean (Kali scale)",      "Laj & Channell 2007 [1]"),
    (4.70e5,  5.0e4,  "Geodynamo oscillation (Kali scale)",           "Constable et al. 2016 [F39]"),
    (1.728e6, 1.0e5,  "Milankovitch super-cycle (Satya/Sandhyā scale)", "Berger & Loutre 1991 [3]"),
    (4.320e6, 5.0e4,  "Extinction sub-cycle 26My÷6 (CY scale)",       "Raup & Sepkoski 1982 [4]"),
    (4.350e9, 3.0e7,  "Hadean crust/water formation (Kalpa scale)",    "Valley et al. 2014 [5,6]"),
    (4.500e9, 1.5e8,  "Earth habitable window midpoint (Manu scale)",  "Rushby et al. 2013 [25]"),
    # Permian — included for completeness, treated separately (F43)
    (2.519e8, 2.4e4,  "Permian extinction (Ardhanārīśvara product)",   "Shen et al. 2011 [38]"),
]

# ── Vedic time values (base=432,000; multipliers from the architecture) ──────
BASE = 432_000  # yr  (Kali Yuga)

# Multiplier set relative to BASE.
# Rationale for each:
#   1     → Kali Yuga itself
#   2     → Dvāpara  (2×Kali)
#   3     → Tretā    (3×Kali)
#   4     → Satya    (4×Kali)
#  10     → Chatur Yuga (1+2+3+4=10 × Kali)
# 710     → Manvantara (71 × CY = 71×10 × Kali)
#10000    → Kalpa    (1000 × CY = 10,000 × Kali)
VEDIC_MULTIPLIERS = [1, 2, 3, 4, 10, 710, 10_000]

VEDIC_VALUES = [BASE * m for m in VEDIC_MULTIPLIERS]
VEDIC_LABELS = [
    "Kali Yuga     (432,000 yr)",
    "Dvāpara Yuga  (864,000 yr)",
    "Tretā Yuga    (1.296 My)",
    "Satya Yuga    (1.728 My)",
    "Chatur Yuga   (4.320 My)",
    "Manvantara    (306.72 My)",
    "Kalpa         (4.320 Gy)",
]

# ── Overlap function ─────────────────────────────────────────────────────────
def overlaps_any(vedic_val, geo_list, threshold=1.0):
    """Return True if vedic_val falls within threshold×sigma of ANY geo comparator."""
    for geo_val, sigma, *_ in geo_list:
        if sigma > 0 and abs(vedic_val - geo_val) / sigma <= threshold:
            return True
    return False


def count_overlaps(vedic_vals, geo_list, threshold=1.0):
    """Count how many vedic_vals overlap at least one geo comparator."""
    return sum(overlaps_any(v, geo_list, threshold) for v in vedic_vals)


# ── Monte Carlo ──────────────────────────────────────────────────────────────
def run_mc(base, vedic_multipliers, geo_list, n_sims=100_000,
           threshold=1.0, rng=None):
    """
    Parameters
    ----------
    base              : float  – fixed base number (432,000)
    vedic_multipliers : list   – the actual Vedic multiplier set
    geo_list          : list   – geophysical comparators
    n_sims            : int    – MC iterations
    threshold         : float  – sigma threshold for overlap (default 1.0)
    rng               : np.random.Generator or None

    Returns
    -------
    dict with keys:
        vedic_overlap_count   : int
        sim_overlap_counts    : np.ndarray shape (n_sims,)
        p_value               : float  (one-tailed: P(random ≥ vedic))
        null_mean             : float
        null_std              : float
        k                     : int    (number of multipliers)
        log_range             : tuple  (log10 min, log10 max)
    """
    if rng is None:
        rng = np.random.default_rng()

    k = len(vedic_multipliers)
    log_min = np.log10(min(vedic_multipliers))
    log_max = np.log10(max(vedic_multipliers))

    vedic_vals = [base * m for m in vedic_multipliers]
    vedic_count = count_overlaps(vedic_vals, geo_list, threshold)

    # Null: draw K multipliers i.i.d. from log-uniform [log_min, log_max]
    log_mults = rng.uniform(log_min, log_max, size=(n_sims, k))
    rand_mults = 10 ** log_mults                      # shape (n_sims, k)
    rand_vals  = base * rand_mults                    # shape (n_sims, k)

    # Count overlaps for each simulation
    sim_counts = np.zeros(n_sims, dtype=int)
    for j in range(k):
        col = rand_vals[:, j]
        for geo_val, sigma, *_ in geo_list:
            if sigma > 0:
                hits = np.abs(col - geo_val) / sigma <= threshold
                sim_counts += hits.astype(int)
    # Note: this double-counts if a single vedic value hits >1 geo.
    # Recompute correctly:
    sim_counts = np.array([
        sum(
            any(abs(rand_vals[i, j] - gv) / gs <= threshold
                for gv, gs, *_ in geo_list if gs > 0)
            for j in range(k)
        )
        for i in range(n_sims)
    ]) if n_sims <= 10000 else _fast_count(rand_vals, geo_list, threshold)

    p_val = np.mean(sim_counts >= vedic_count)

    return dict(
        vedic_overlap_count = vedic_count,
        sim_overlap_counts  = sim_counts,
        p_value             = p_val,
        null_mean           = sim_counts.mean(),
        null_std            = sim_counts.std(),
        k                   = k,
        log_range           = (log_min, log_max),
    )


def _fast_count(rand_vals, geo_list, threshold):
    """Vectorised overlap count across all simulations."""
    n_sims, k = rand_vals.shape
    any_hit = np.zeros((n_sims, k), dtype=bool)
    for geo_val, sigma, *_ in geo_list:
        if sigma > 0:
            any_hit |= (np.abs(rand_vals - geo_val) / sigma <= threshold)
    return any_hit.sum(axis=1)


# ── Second test: fix multipliers, randomise base ─────────────────────────────
def run_mc_base(vedic_multipliers, geo_list, n_sims=100_000,
                threshold=1.0, base_range=(1e4, 1e7), rng=None):
    """Hold multipliers fixed, draw base from log-uniform [base_range]."""
    if rng is None:
        rng = np.random.default_rng()

    vedic_vals   = [BASE * m for m in vedic_multipliers]
    vedic_count  = count_overlaps(vedic_vals, geo_list, threshold)

    log_lo, log_hi = np.log10(base_range[0]), np.log10(base_range[1])
    rand_bases  = 10 ** rng.uniform(log_lo, log_hi, size=n_sims)
    rand_vals   = rand_bases[:, None] * np.array(vedic_multipliers)  # (n_sims, k)

    sim_counts  = _fast_count(rand_vals, geo_list, threshold)
    p_val       = np.mean(sim_counts >= vedic_count)

    return dict(
        vedic_overlap_count = vedic_count,
        p_value             = p_val,
        null_mean           = sim_counts.mean(),
        null_std            = sim_counts.std(),
        base_range          = base_range,
    )


# ── Exhaustive pairwise table ────────────────────────────────────────────────
def exhaustive_table(vedic_vals, vedic_labels, geo_list, threshold=1.0):
    """Return a list of dicts for every (Vedic, Geo) pair."""
    rows = []
    for vv, vl in zip(vedic_vals, vedic_labels):
        best_gap_pct = None
        best_sigma   = None
        best_geo     = None
        for gv, gs, gl, gs_src in geo_list:
            gap_pct = abs(vv - gv) / gv * 100
            sigma_n  = abs(vv - gv) / gs if gs > 0 else float('inf')
            rows.append(dict(
                vedic_label = vl,
                vedic_val   = vv,
                geo_label   = gl,
                geo_val     = gv,
                geo_sigma   = gs,
                geo_source  = gs_src,
                gap_pct     = gap_pct,
                sigma_n     = sigma_n,
                overlap     = sigma_n <= threshold,
            ))
    return rows


# ── Report ───────────────────────────────────────────────────────────────────
def print_report(mc_mult, mc_base, exhaustive):
    print("=" * 70)
    print("VEDIC TIME CYCLES — MONTE CARLO REPORT")
    print("=" * 70)

    print("\n[ TEST 1 ] Fix base=432,000; randomise multiplier set")
    print(f"  Vedic overlap count  : {mc_mult['vedic_overlap_count']}")
    print(f"  Null mean ± std      : {mc_mult['null_mean']:.2f} ± {mc_mult['null_std']:.2f}")
    print(f"  p-value (one-tailed) : {mc_mult['p_value']:.4f}")
    print(f"  Multiplier log-range : 10^{mc_mult['log_range'][0]:.2f} – 10^{mc_mult['log_range'][1]:.2f}")
    print(f"  K multipliers        : {mc_mult['k']}")

    print("\n[ TEST 2 ] Fix multipliers; randomise base (log-uniform 1e4–1e7 yr)")
    print(f"  Vedic overlap count  : {mc_base['vedic_overlap_count']}")
    print(f"  Null mean ± std      : {mc_base['null_mean']:.2f} ± {mc_base['null_std']:.2f}")
    print(f"  p-value (one-tailed) : {mc_base['p_value']:.4f}")

    print("\n[ EXHAUSTIVE PAIRWISE TABLE ]")
    print(f"{'Vedic':<30} {'Geo comparator':<42} {'Gap%':>6} {'σ':>6} {'Hit?':>5}")
    print("-" * 96)
    prev_vedic = ""
    for r in exhaustive:
        vl = r['vedic_label'] if r['vedic_label'] != prev_vedic else ""
        prev_vedic = r['vedic_label']
        hit = "✓" if r['overlap'] else "✗"
        print(f"{vl:<30} {r['geo_label']:<42} {r['gap_pct']:>5.1f}% {r['sigma_n']:>5.1f}σ {hit:>5}")

    print("\n[ SUMMARY ]")
    total_vedic = len(set(r['vedic_label'] for r in exhaustive))
    vedic_any_hit = sum(
        1 for vl in set(r['vedic_label'] for r in exhaustive)
        if any(r['overlap'] for r in exhaustive if r['vedic_label'] == vl)
    )
    print(f"  Vedic constants with ≥1 geophysical overlap : {vedic_any_hit}/{total_vedic}")
    print(f"  Total pairwise overlaps (1σ)                : {sum(r['overlap'] for r in exhaustive)}")


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Vedic MC simulation")
    parser.add_argument("--n",    type=int, default=100_000, help="MC iterations")
    parser.add_argument("--seed", type=int, default=2026,    help="RNG seed")
    parser.add_argument("--threshold", type=float, default=1.0, help="σ threshold")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── PRE-SPECIFIED geo list (textually justified, specified before measurement)
    GEO_PRE = [
        (4.50e5, 5.0e4, "Geomagnetic reversal mean (450 ky)",          "Laj & Channell 2007"),
        (4.70e5, 5.0e4, "Geodynamo oscillation (~470 ky)",             "Constable et al. 2016"),
        (1.70e6, 1.0e5, "Milankovitch super-cycle B (1.7 My)",         "Berger & Loutre 1991"),
        (4.333e6, 1.667e5, "26My/6 extinction sub-cycle (4.33 My)",    "Raup & Sepkoski 1982"),
        (4.35e9, 3.0e7,  "Hadean crust/water (4.35 Ga)",               "Valley et al. 2014"),
        (4.50e9, 5.0e8,  "Earth habitable midpoint (7th Manu)",         "Rushby et al. 2013"),
    ]
    mc_pre = run_mc(BASE, VEDIC_MULTIPLIERS, GEO_PRE,
                    n_sims=args.n, threshold=args.threshold, rng=rng)
    print("\n[ PRE-SPECIFIED LIST MC (textually justified comparators — for paper) ]")
    print(f"  Vedic overlap count  : {mc_pre['vedic_overlap_count']}")
    print(f"  Null mean ± std      : {mc_pre['null_mean']:.2f} ± {mc_pre['null_std']:.2f}")
    print(f"  p-value (one-tailed) : {mc_pre['p_value']:.5f}")
    print()

    # ── Exhaustive geo list (ALL major periodicities)
    GEO_ALL = [
        # ── Milankovitch ──────────────────────────────────────────────────
        (2.30e4,  2.0e3,  "Milankovitch precession (23 ky)",             "Berger 1978"),
        (4.10e4,  2.0e3,  "Milankovitch obliquity (41 ky)",              "Berger 1978"),
        (1.00e5,  1.0e4,  "Milankovitch eccentricity short (100 ky)",    "Hays et al. 1976"),
        (4.13e5,  1.0e4,  "Milankovitch eccentricity long (413 ky)",     "Berger & Loutre 1991"),
        # ── Geomagnetic ──────────────────────────────────────────────────
        (4.50e5,  5.0e4,  "Geomagnetic reversal mean (450 ky)",          "Laj & Channell 2007"),
        (4.70e5,  5.0e4,  "Geodynamo oscillation (~470 ky)",             "Constable et al. 2016"),
        # ── Milankovitch super-cycles ─────────────────────────────────────
        (1.20e6,  1.0e5,  "Milankovitch super-cycle A (1.2 My)",         "Berger & Loutre 1991"),
        (1.70e6,  1.0e5,  "Milankovitch super-cycle B (1.7 My)",         "Berger & Loutre 1991"),
        (2.40e6,  2.0e5,  "Milankovitch super-cycle C (2.4 My)",         "Berger & Loutre 1991"),
        # ── Extinction cycles ─────────────────────────────────────────────
        (2.60e7,  1.0e6,  "Extinction periodicity (26 My)",              "Raup & Sepkoski 1982"),
        (6.20e7,  3.0e6,  "Extinction periodicity (62 My)",              "Rohde & Muller 2005"),
        # ── Galactic / tectonic ───────────────────────────────────────────
        (2.30e8,  2.0e7,  "Galactic year / mantle overturn (230 My)",    "Leitch & Vasisht 1998"),
        (5.00e8,  1.0e8,  "Wilson cycle (500 My)",                       "Wilson 1966"),
        # ── Deep time ────────────────────────────────────────────────────
        (2.519e8, 2.4e4,  "Permian extinction (251.9 Ma)",               "Shen et al. 2011"),
        (5.41e8,  1.0e6,  "Cambrian explosion (541 Ma)",                 "Gradstein et al. 2012"),
        (7.00e8,  5.0e7,  "Snowball Earth (700 Ma)",                     "Hoffman et al. 1998"),
        (2.40e9,  1.0e8,  "Great Oxygenation Event (2.4 Ga)",            "Holland 2006"),
        (4.35e9,  3.0e7,  "Hadean crust/water (4.35 Ga)",               "Valley et al. 2014"),
        (4.54e9,  5.0e7,  "Earth total age (4.54 Ga)",                   "Patterson 1956"),
        (4.50e9,  5.0e8,  "Earth habitable midpoint (7th Manu)",         "Rushby et al. 2013"),
    ]

    print(f"Running with N={args.n:,} simulations, seed={args.seed}, threshold={args.threshold}σ\n")

    mc_mult = run_mc(BASE, VEDIC_MULTIPLIERS, GEO_ALL,
                     n_sims=args.n, threshold=args.threshold, rng=rng)

    mc_base = run_mc_base(VEDIC_MULTIPLIERS, GEO_ALL,
                          n_sims=args.n, threshold=args.threshold,
                          base_range=(1e4, 1e7), rng=rng)

    ex = exhaustive_table(VEDIC_VALUES, VEDIC_LABELS, GEO_ALL, threshold=args.threshold)

    print_report(mc_mult, mc_base, ex)

    # Machine-readable output for paper table
    print("\n[ JSON-READY RESULTS ]")
    import json
    out = {
        "mc_multiplier_test": {
            "vedic_overlap_count": mc_mult["vedic_overlap_count"],
            "null_mean": round(mc_mult["null_mean"], 3),
            "null_std": round(mc_mult["null_std"], 3),
            "p_value": mc_mult["p_value"],
            "n_sims": args.n,
            "seed": args.seed,
        },
        "mc_base_test": {
            "vedic_overlap_count": mc_base["vedic_overlap_count"],
            "null_mean": round(mc_base["null_mean"], 3),
            "null_std": round(mc_base["null_std"], 3),
            "p_value": mc_base["p_value"],
            "n_sims": args.n,
        },
        "exhaustive_pairs": [
            {k: (round(v, 4) if isinstance(v, float) else v)
             for k, v in r.items() if k != "sim_overlap_counts"}
            for r in ex
        ]
    }
    print(json.dumps({"summary": {
        "mc_mult_p": mc_mult["p_value"],
        "mc_base_p": mc_base["p_value"],
        "vedic_overlap_count": mc_mult["vedic_overlap_count"],
        "null_mean": round(mc_mult["null_mean"], 2),
    }}, indent=2))


if __name__ == "__main__":
    main()
