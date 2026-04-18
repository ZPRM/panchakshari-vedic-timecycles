# Pañcākṣarī Vedic Time Cycles — Code Repository

Companion code for:

> Chilakalapudi, P. (2026). *Vedic Cosmological Time Cycles and Geophysical Periodicities: A Structural Alignment Study.* Paper 1 v10.
> Zenodo: https://doi.org/10.5281/zenodo.19026517

---

## Repository Contents

| File | Description |
|---|---|
| `VedicCosmological_TimeCycles_and_Geophysical_Periodicities.py` | Original v8.6 analysis (log-uniform null, N=50,000). Full exhaustive scan. |
| `corrected_monte_carlo_v10.py` | **v10 corrected analysis** (empirical null, N=100,000). Primary result for Paper 1 v10. |
| `vedic_paper_figures.py` | Reproduces all figures (Figures 1–5) from the paper. |
| `Panchakshari_Paper_CSALT_Verify_v5.ipynb` | C-SALT Sanskrit verification notebook. Table 1 root verification (23/24 confirmed). |

---

## Quick Start

```bash
pip install numpy matplotlib
```

---

## v8.6 Analysis (Original Pre-Specified)

```bash
python VedicCosmological_TimeCycles_and_Geophysical_Periodicities.py --n 50000 --seed 2026
```

**Expected output (pre-specified list):**
```
[ PRE-SPECIFIED LIST MC (v8.6 analysis) ]
  Vedic overlap count  : 4
  Null mean +/- std    : 0.35 +/- 0.58
  p-value (one-tailed) : 0.00008
```

> **Note:** This is the v8.6 exploratory analysis. It includes comparators subsequently
> corrected in v10 (see Paper 1 §4.1): the Chatur Yuga ÷6 row [REMOVED — CONSTRUCTED]
> and the Tretā/Satya Yuga rows [PC-DERIVED — EXPLORATORY ONLY].

---

## v10 Corrected Analysis (Primary Result — Paper 1 v10)

```bash
python corrected_monte_carlo_v10.py
```

**Expected output:**
```
PRIMARY RESULT (for Paper 1 §4.1):
  2 of 3 testable constants match at 1σ
  Null B: 24 of 100,000 simulations >= 2 hits
  p = 0.00024
  Note: Kalpa match is post-hoc reframed (§6.2).
  Without reframe: 1 of 3 (Kali Yuga only).
```

**Corrections applied (v8.6 → v10):**
1. Chatur Yuga `26 My ÷ 6` comparator removed — divisor 6 is the match condition, not an observed geophysical quantity
2. Tretā Yuga (1.296 My) and Satya Yuga (1.728 My) comparators reclassified `[PC-DERIVED]` — computed from Berger & Loutre (1991) orbital parameter dataset, not named periodicities in that publication
3. Kali Yuga counted once — Constable et al. 2016 reported as secondary observation only
4. Null distribution: log-uniform → empirical integer pool (`VEDIC_MULTIPLIER_POOL`, 21 integers from Vedic, Babylonian, Mesoamerican, Chinese cosmologies)

---

## Empirical Null Pool

`VEDIC_MULTIPLIER_POOL` (cited in Paper 1 §4.3) is defined in both scripts:

```python
VEDIC_MULTIPLIER_POOL = [
    4, 3, 2, 1, 71, 14, 1000,   # Vedic
    60, 12, 30, 6, 4,            # Babylonian
    20, 18, 13, 52, 365,         # Mesoamerican
    60, 12, 10, 4,               # Chinese
]
```

21 integers total, sampled with replacement, 5 per simulation, applied cumulatively to base 432,000 yr.

---

## Replication Parameters

| Parameter | v8.6 | v10 corrected |
|---|---|---|
| N simulations | 50,000 | 100,000 |
| Seed | 2026 | 2026 |
| Base | 432,000 yr | 432,000 yr |
| Null | Log-uniform [10⁰, 10⁴] | Empirical integer pool |
| Constants tested | 7 | 3 (after corrections) |
| Primary p-value | 0.00008 | 0.00024 |

All results are deterministic given the seed. No proprietary data used.

---

## Sanskrit Verification

```bash
# Requires Google Colab or Jupyter + requests
# Run Panchakshari_Paper_CSALT_Verify_v5.ipynb
```

Verifies Table 1 Śivopāsanā epithet root decompositions against three dictionaries via C-SALT REST API (Monier-Williams, Apte, Böhtlingk-Roth). Result: 23 of 24 roots confirmed in ≥1 dictionary (one SLP1 encoding ambiguity for `kaṇṭha`).

---

## Changelog

**v10.0 — 18 April 2026**
- Added `corrected_monte_carlo_v10.py` with corrected exploratory analysis
- Added `VEDIC_MULTIPLIER_POOL` constant to main script (cited in Paper 1 §4.3)
- Annotated `[PC-DERIVED]` and `[REMOVED]` comparators in `GEO_PRE` list
- Repository made public

**v8.6 — 14 March 2026**
- Original pre-specified analysis submitted with Paper 1 v8.6
- Archived at Zenodo: https://doi.org/10.5281/zenodo.19026517

---

## Citation

```bibtex
@misc{chilakalapudi2026vedic,
  author    = {Chilakalapudi, Pramod},
  title     = {Vedic Cosmological Time Cycles and Geophysical Periodicities:
               A Structural Alignment Study},
  year      = {2026},
  doi       = {10.5281/zenodo.19026517},
  publisher = {Zenodo}
}
```

---

## Author

Pramod Chilakalapudi  
Independent Researcher, Eindhoven, Netherlands  
ORCID: [0009-0008-2646-5822](https://orcid.org/0009-0008-2646-5822)
