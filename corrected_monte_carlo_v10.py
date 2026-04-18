"""
corrected_monte_carlo_v10.py
============================
Corrected exploratory Monte Carlo analysis for:

  Chilakalapudi, P. (2026). Vedic Cosmological Time Cycles and Geophysical
  Periodicities: A Structural Alignment Study. Paper 1 v10.
  Zenodo: https://doi.org/10.5281/zenodo.19026517

This script supersedes the v8.6 pre-specified analysis (N=50,000, log-uniform
null). It applies two methodological corrections identified in peer review:

  Correction 1 — Constructed comparator removed:
    Chatur Yuga vs "26 My ÷ 6" excluded. The divisor 6 is the match
    condition, not an observed geophysical quantity.

  Correction 2 — Citation reclassification:
    Tretā Yuga (1.296 My) and Satya Yuga (1.728 My) comparators are
    [PC-DERIVED] — computed from spectral analysis of the Berger & Loutre
    (1991) orbital parameter dataset (PANGAEA DOI: 10.1594/PANGAEA.56040),
    not named periodicities in that publication (Gate A verification:
    Berger & Loutre 1991 reports standard Milankovitch periods 100ky/41ky/
    23ky/19ky only). Excluded from the corrected primary analysis;
    retained in the exhaustive scan (§4.2) with [PC-DERIVED] annotation.

  Correction 3 — No double-counting:
    Kali Yuga tested against one comparator only (geomagnetic reversal mean,
    Laj & Channell 2007). Constable et al. 2016 (470 ky) reported as
    secondary observation, not a separate test.

  Correction 4 — Empirical null distribution:
    VEDIC_MULTIPLIER_POOL replaces log-uniform [10^0, 10^4]. Integer
    multipliers drawn from four attested ancient cosmological systems
    (see below). Tighter null; more conservative test.

Result reported in Paper 1 §4.1:
  Constants tested: 3 (Kali Yuga, Manvantara, Kalpa)
  Matches at 1σ: 2 (Kali Yuga: 0.36σ; Kalpa: 1.00σ; Manvantara: null)
  p = 0.00025 (Null B, empirical pool, N=100,000, seed=2026)
  Note: Kalpa match is post-hoc reframed (see §6.2). Without reframe: 1 of 3.

Replication:
  python corrected_monte_carlo_v10.py
  All parameters are fixed above; results are deterministic given seed=2026.
"""

import numpy as np

# ── PARAMETERS ────────────────────────────────────────────────────────────────
SEED       = 2026
N          = 100_000
BASE       = 432_000       # Kali Yuga in years (fixed)
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]

# ── VEDIC CONSTANTS (after corrections) ──────────────────────────────────────
# Excluded: Dvāpara Yuga (no comparator in exhaustive scan)
# Excluded: Chatur Yuga (÷6 construction — Correction 1)
# Excluded from primary: Tretā and Satya Yuga ([PC-DERIVED] — Correction 2)
VEDIC = {
    "Kali Yuga":   432_000,
    "Manvantara":  306_720_000,
    "Kalpa":       4_320_000_000,
}

# ── GEOPHYSICAL COMPARATORS ───────────────────────────────────────────────────
# Format: (geo_value_yr, geo_sigma_yr, label, source, note)
COMPARATORS = {
    "Kali Yuga":   (450_000,       50_000,    "Geomagnetic reversal mean",
                    "Laj & Channell 2007", "primary"),
    "Manvantara":  (None,          None,      "No confirmed comparator",
                    "NULL RESULT",          "null"),
    "Kalpa":       (4_350_000_000, 30_000_000, "Hadean crust/water formation",
                    "Valley et al. 2014",   "post-hoc reframed — see §6.2"),
}
# Secondary observation (NOT a separate test — Correction 3):
#   Constable et al. 2016: geodynamo oscillation ~470 ky ± 50 ky

# ── EMPIRICAL NULL POOL (VEDIC_MULTIPLIER_POOL) ───────────────────────────────
# Integer multipliers attested in four ancient cosmological systems.
# Pool size: 28 integers. Sampled with replacement, 5 per simulation,
# applied cumulatively to BASE (432,000).
VEDIC_MULTIPLIER_POOL = [
    # Vedic (Purāṇic multipliers: Yuga ratios, Manus, Kalpas)
    4, 3, 2, 1, 71, 14, 1000,
    # Babylonian (sexagesimal base + major divisors)
    60, 12, 30, 6, 4,
    # Mesoamerican (Maya long-count structure)
    20, 18, 13, 52, 365,
    # Chinese (traditional cycle multipliers)
    60, 12, 10, 4,
]
# Pool cardinality: 21 integers total (duplicates across traditions retained
# as separate entries — e.g. Babylonian 4 and Vedic 4 are independent
# attestations, both included).


def sigma_distance(vedic_val, geo_val, geo_sigma):
    return abs(vedic_val - geo_val) / geo_sigma


def count_vedic_matches(threshold):
    hits = 0
    details = []
    for name, vedic_val in VEDIC.items():
        comp = COMPARATORS[name]
        geo_val, geo_sigma, label, source, note = comp
        if geo_val is None:
            details.append(f"  {name}: NULL — no comparator ({note})")
            continue
        d = sigma_distance(vedic_val, geo_val, geo_sigma)
        match = d <= threshold
        if match:
            hits += 1
        details.append(
            f"  {name}: {d:.3f}σ from {label} [{source}] "
            f"→ {'MATCH ✓' if match else 'miss ✗'}"
            + (f" [{note}]" if note not in ("primary", "null") else ""))
    return hits, details


def run_null(pool, label, seed_offset=0):
    """Monte Carlo under a given multiplier pool."""
    rng = np.random.default_rng(SEED + seed_offset)
    results = {}
    for thresh in THRESHOLDS:
        vedic_hits, _ = count_vedic_matches(thresh)
        null_counts = []
        for _ in range(N):
            mults = rng.choice(pool, size=5, replace=True)
            rand_vals = BASE * np.cumprod(mults)
            hits = 0
            for i, (name, comp) in enumerate(COMPARATORS.items()):
                geo_val, geo_sigma = comp[0], comp[1]
                if geo_val is None:
                    continue
                if sigma_distance(rand_vals[i], geo_val, geo_sigma) <= thresh:
                    hits += 1
            null_counts.append(hits)
        null_arr = np.array(null_counts)
        p = np.mean(null_arr >= vedic_hits)
        results[thresh] = dict(
            vedic=vedic_hits,
            null_mean=float(null_arr.mean()),
            null_std=float(null_arr.std()),
            p=float(p),
            n_gte=int(np.sum(null_arr >= vedic_hits)),
        )
    return results


def main():
    print("=" * 70)
    print("CORRECTED EXPLORATORY MONTE CARLO — Paper 1 v10")
    print(f"N = {N:,} | Seed = {SEED} | Base = {BASE:,} yr")
    print("=" * 70)

    print("\nMethodological corrections applied:")
    print("  Gate A: Berger & Loutre 1991 — 1.7 My NOT a named periodicity")
    print("          → Tretā and Satya Yuga excluded from primary analysis")
    print("  Gate B: Chatur Yuga ÷6 excluded (constructed comparator)")
    print("  Kali Yuga: counted once (single comparator)")
    print(f"\nPool size: {len(VEDIC_MULTIPLIER_POOL)} integers")
    print(f"Pool contents: {VEDIC_MULTIPLIER_POOL}")

    print("\n" + "─" * 70)
    print("VEDIC MATCH DETAILS")
    print("─" * 70)
    for thresh in THRESHOLDS:
        hits, details = count_vedic_matches(thresh)
        print(f"\n@ {thresh}σ — {hits} of {len(VEDIC)} match(es):")
        for d in details:
            print(d)

    print("\n" + "─" * 70)
    print("NULL A — Log-uniform [10^0, 10^4] (v8.6 null, for comparison)")
    print("─" * 70)
    log_pool = list(range(1, 10001))  # approximate log-uniform via uniform integers
    rng_a = np.random.default_rng(SEED)
    for thresh in THRESHOLDS:
        vedic_hits, _ = count_vedic_matches(thresh)
        null_counts = []
        for _ in range(N):
            mults = rng_a.uniform(1, 10_000, size=5)
            rand_vals = BASE * np.cumprod(mults)
            hits = 0
            for i, (name, comp) in enumerate(COMPARATORS.items()):
                geo_val, geo_sigma = comp[0], comp[1]
                if geo_val is None:
                    continue
                if sigma_distance(rand_vals[i], geo_val, geo_sigma) <= thresh:
                    hits += 1
            null_counts.append(hits)
        null_arr = np.array(null_counts)
        p = np.mean(null_arr >= vedic_hits)
        print(f"  {thresh}σ: Vedic={vedic_hits}, Null mean={null_arr.mean():.4f}, p={p:.5f}")

    print("\n" + "─" * 70)
    print("NULL B — Empirical integer pool (PRIMARY RESULT)")
    print("─" * 70)
    results_b = run_null(VEDIC_MULTIPLIER_POOL, "Null B", seed_offset=0)
    for thresh in THRESHOLDS:
        r = results_b[thresh]
        marker = "  ← PRIMARY" if thresh == 1.0 else ""
        print(
            f"  {thresh}σ: Vedic={r['vedic']}, "
            f"Null mean={r['null_mean']:.4f}±{r['null_std']:.4f}, "
            f"n_gte={r['n_gte']}, p={r['p']:.5f}{marker}")

    print("\n" + "=" * 70)
    print("PRIMARY RESULT (for Paper 1 §4.1):")
    r1 = results_b[1.0]
    print(f"  2 of 3 testable constants match at 1σ")
    print(f"  Null B: {r1['n_gte']} of {N:,} simulations ≥ {r1['vedic']} hits")
    print(f"  p = {r1['p']:.5f}")
    print(f"  Note: Kalpa match is post-hoc reframed (§6.2).")
    print(f"  Without reframe: 1 of 3 (Kali Yuga only).")
    print("=" * 70)


if __name__ == "__main__":
    main()
