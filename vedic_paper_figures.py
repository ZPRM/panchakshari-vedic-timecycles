"""
vedic_paper_figures.py
======================
Generates all 5 publication-quality figures for:
"Vedic Cosmological Time Cycles and Geophysical Periodicities"
Author: Pramod Chilakalapudi

Figures produced:
  fig1_null_distribution.png  — MC null distribution histogram
  fig2_timeline.png           — Log-scale Vedic vs Geo timeline
  fig3_sigma_bars.png         — Sigma distance bar chart
  fig4_mp_c11_structure.png   — Mantra Pushpam + C11 dual diagram
  fig5_universe_age.png       — Universe age comparison (Appendix A.3)

Usage:
  python vedic_paper_figures.py

Requirements: numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import warnings
warnings.filterwarnings('ignore')

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

BLUE   = '#1F4E79'
GOLD   = '#8B6914'
RED    = '#8B1A1A'
GREEN  = '#1A5C1A'
GREY   = '#555555'
LBLUE  = '#D6E4F0'
LGOLD  = '#FFF3CD'

# ── Figure 1: NULL DISTRIBUTION HISTOGRAM ─────────────────────────────────────
def make_fig1():
    rng = np.random.default_rng(2026)
    BASE = 432_000

    GEO_PRE = [
        (4.50e5, 5.0e4), (4.70e5, 5.0e4), (1.70e6, 1.0e5),
        (4.333e6, 1.667e5), (4.35e9, 3.0e7), (4.50e9, 5.0e8),
    ]
    VEDIC_MULTS = [1, 2, 3, 4, 10, 710, 10000]
    VEDIC_VALS  = [BASE * m for m in VEDIC_MULTS]

    def count_ov(vals):
        return sum(
            any(abs(v - gv) / gs <= 1.0 for gv, gs in GEO_PRE)
            for v in vals
        )

    vedic_count = count_ov(VEDIC_VALS)

    N = 50_000
    log_mults = rng.uniform(0, 4, size=(N, 7))
    rand_mults = 10 ** log_mults
    rand_vals  = BASE * rand_mults
    any_hit = np.zeros((N, 7), dtype=bool)
    for gv, gs in GEO_PRE:
        any_hit |= (np.abs(rand_vals - gv) / gs <= 1.0)
    sim_counts = any_hit.sum(axis=1)
    p_val = np.mean(sim_counts >= vedic_count)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(-0.5, sim_counts.max() + 1.5)
    counts, edges, patches = ax.hist(sim_counts, bins=bins, color=LBLUE,
                                      edgecolor=BLUE, linewidth=0.8, zorder=2)

    # Colour bars >= vedic_count in gold
    for i, patch in enumerate(patches):
        left = patch.get_x()
        if left + 0.5 >= vedic_count:
            patch.set_facecolor(LGOLD)
            patch.set_edgecolor(GOLD)

    # Vertical line at Vedic count
    ax.axvline(vedic_count, color=RED, linewidth=2.2, zorder=3,
               label=f'Vedic count = {vedic_count}')

    # Annotations
    ax.annotate(
        f'Vedic architecture:\n{vedic_count} overlaps\n(p = {p_val:.5f})',
        xy=(vedic_count, counts[vedic_count] if vedic_count < len(counts) else 50),
        xytext=(vedic_count + 0.6, counts.max() * 0.6),
        fontsize=10, color=RED,
        arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0', edgecolor=RED, alpha=0.9)
    )

    ax.set_xlabel('Number of geophysical overlaps (1σ criterion)', labelpad=8)
    ax.set_ylabel('Frequency (out of 50,000 simulations)', labelpad=8)
    ax.set_title(
        'Figure 1. Monte Carlo Null Distribution\n'
        'Random multiplier sets vs. pre-specified geophysical comparators (N = 50,000; seed 2026)',
        pad=12
    )
    ax.set_xlim(-0.5, sim_counts.max() + 0.5)

    null_mean = sim_counts.mean()
    ax.axvline(null_mean, color=GREY, linewidth=1.2, linestyle=':', zorder=2)
    ax.text(null_mean + 0.05, counts.max() * 0.95,
            f'Null mean = {null_mean:.2f}', fontsize=9, color=GREY)

    legend_patches = [
        mpatches.Patch(facecolor=LBLUE, edgecolor=BLUE, label='Null distribution'),
        mpatches.Patch(facecolor=LGOLD, edgecolor=GOLD, label='≥ Vedic count (shaded)'),
        Line2D([0], [0], color=RED, linewidth=2, label=f'Vedic count = {vedic_count}'),
    ]
    ax.legend(handles=legend_patches, loc='upper right', fontsize=9,
              framealpha=0.9, edgecolor=GREY)

    fig.text(0.5, -0.02,
             'Null: log-uniform multiplier sets, same range as Vedic (10⁰–10⁴), base = 432,000 yr fixed.',
             ha='center', fontsize=8.5, color=GREY, style='italic')

    plt.tight_layout()
    plt.savefig('/home/claude/fig1_null_distribution.png')
    plt.close()
    print('Fig 1 saved.')


# ── Figure 2: TIMELINE PLOT ────────────────────────────────────────────────────
def make_fig2():
    # All data
    vedic = [
        ('Kali Yuga',    4.32e5),
        ('Dvāpara Yuga', 8.64e5),
        ('Tretā Yuga',   1.296e6),
        ('Satya Yuga',   1.728e6),
        ('Chatur Yuga',  4.32e6),
        ('Manvantara',   3.0672e8),
        ('Kalpa',        4.32e9),
    ]

    geo = [
        # (label, value, sigma, matched_vedic_idx)
        ('Geomag. reversal\nmean (450 ky)',    4.50e5, 5.0e4, 0),
        ('Geodynamo\nosc. (~470 ky)',           4.70e5, 5.0e4, 0),
        ('Milankovitch\nsuper-cycle B (1.7 My)', 1.70e6, 1.0e5, 3),
        ('26 My÷6\nextinction sub (4.33 My)',   4.333e6, 1.667e5, 4),
        ('Hadean crust/\nwater (4.35 Ga)',       4.35e9, 3.0e7, 6),
        ('Earth habitable\nmidpoint (4.5 Ga)',   4.50e9, 5.0e8, 6),
        # Misses (no match)
        ('Galactic year\n(230 My) — NO MATCH',   2.30e8, 2.0e7, 5),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xscale('log')

    # Plot geophysical error bars
    for i, (label, val, sig, vi) in enumerate(geo):
        is_miss = 'NO MATCH' in label
        color = RED if is_miss else GREEN

        # Error bar shading
        ax.axvspan(val - sig, val + sig, alpha=0.12, color=color, zorder=1)
        ax.axvline(val, color=color, linewidth=1.0, linestyle='--', alpha=0.6, zorder=2)

    # Plot Vedic values as vertical coloured lines with labels
    vedic_colors = [BLUE if i not in [1, 5] else RED for i in range(7)]
    # 1=Dvapara (miss), 5=Manvantara (miss)
    vedic_colors = [BLUE, RED, BLUE, BLUE, BLUE, RED, BLUE]

    y_offsets = [0.92, 0.80, 0.68, 0.56, 0.44, 0.32, 0.20]
    for i, ((name, val), color) in enumerate(zip(vedic, vedic_colors)):
        ax.axvline(val, color=color, linewidth=2.2, zorder=4)
        ax.text(val * 1.05, y_offsets[i], name,
                fontsize=8.5, color=color, va='center',
                transform=ax.get_xaxis_transform(),
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=color, alpha=0.85))

    ax.set_xlim(1e5, 2e10)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Time (years) — logarithmic scale', labelpad=8)
    ax.set_title(
        'Figure 2. Vedic Time Constants vs. Geophysical Periodicities\n'
        'Blue lines = Vedic constants with ≥1σ match  |  Red lines = no match  |  '
        'Green shading = 1σ geophysical range',
        pad=12
    )

    # X-axis tick labels
    ax.set_xticks([1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
    ax.set_xticklabels(['100 ky', '1 My', '10 My', '100 My', '1 Gy', '10 Gy'])

    legend_elems = [
        Line2D([0], [0], color=BLUE, linewidth=2.5, label='Vedic constant — match found'),
        Line2D([0], [0], color=RED,  linewidth=2.5, label='Vedic constant — NO MATCH'),
        mpatches.Patch(facecolor=GREEN, alpha=0.25, label='Geophysical ±1σ range (match)'),
        mpatches.Patch(facecolor=RED,   alpha=0.15, label='Geophysical ±1σ range (no match)'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor=GREY)

    plt.tight_layout()
    plt.savefig('/home/claude/fig2_timeline.png')
    plt.close()
    print('Fig 2 saved.')


# ── Figure 3: SIGMA DISTANCE BAR CHART ────────────────────────────────────────
def make_fig3():
    # Best (Vedic, Geo) pairs — one per Vedic constant
    pairs = [
        ('Kali Yuga (432 ky)\nvs. Geomag. reversal',       0.36,  True),
        ('Dvāpara Yuga (864 ky)\nvs. closest geo (no match)', 8.28, False),
        ('Tretā Yuga (1.296 My)\nvs. Milankovitch A',           0.96, True),   # 0.96σ genuine hit
        ('Satya Yuga (1.728 My)\nvs. Milankovitch B',       0.28,  True),
        ('Chatur Yuga (4.32 My)\nvs. closest exhaustive (5.0σ)', 5.0, False),  # pre-spec only
        ('Manvantara (306 My)\nvs. Galactic year (closest)', 3.84, False),
        ('Kalpa (4.32 Gy)\nvs. Hadean crust',               1.00,  True),
    ]

    labels = [p[0] for p in pairs]
    sigmas = [p[1] for p in pairs]
    hits   = [p[2] for p in pairs]

    fig, ax = plt.subplots(figsize=(9, 6))

    colors = [GREEN if h else RED for h in hits]
    # Tretā is a genuine hit (0.96σ)
    # colors[2] remains GREEN

    bars = ax.barh(range(len(pairs)), sigmas, color=colors, alpha=0.75,
                   edgecolor='white', linewidth=0.5, height=0.6)

    # 1-sigma line
    ax.axvline(1.0, color=BLUE, linewidth=2.0, linestyle='--', zorder=3,
               label='1σ threshold')

    # Value labels
    for i, (bar, s) in enumerate(zip(bars, sigmas)):
        ax.text(s + 0.06, i, f'{s:.2f}σ', va='center', fontsize=9.5,
                color=colors[i])

    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Sigma distance from best geophysical match', labelpad=8)
    ax.set_title(
        'Figure 3. Sigma Distance: Each Vedic Constant vs. Best Geophysical Match\n'
        'Green = within 1σ (match)  |  Gold = borderline  |  Red = no match',
        pad=12
    )
    ax.set_xlim(0, max(sigmas) * 1.18)

    legend_elems = [
        mpatches.Patch(facecolor=GREEN, alpha=0.75, label='1σ match'),
        
        mpatches.Patch(facecolor=RED,   alpha=0.75, label='No match'),
        Line2D([0], [0], color=BLUE, linewidth=2, linestyle='--', label='1σ threshold'),
    ]
    ax.legend(handles=legend_elems, loc='lower right', fontsize=9,
              framealpha=0.9, edgecolor=GREY)

    plt.tight_layout()
    plt.savefig('/home/claude/fig3_sigma_bars.png')
    plt.close()
    print('Fig 3 saved.')


# ── Figure 4: MP CHAIN + C11 DUAL DIAGRAM ─────────────────────────────────────
def make_fig4():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 8))

    # LEFT: Mantra Pushpam ascending chain
    ax1.axis('off')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(-0.5, 10)

    chain = [
        ('1. Āpas', 'Primordial waters', '#1E6B9C', False),
        ('2. Candramās', 'Moon = waters', '#1E6B9C', False),
        ('3. Parjanya', 'Rain cloud = waters', '#1E6B9C', False),
        ('4. Samvatsara', 'Year = Prajāpati  ★', '#8B1A1A', True),
        ('5. Āditya', 'Sun = Year  ★', '#8B1A1A', True),
        ('6. Nakṣatrāṇi', 'Stars (light in waters)', '#2E7A2E', False),
        ('7. Vāyu', 'Wind = waters', '#2E7A2E', False),
        ('8. Ākāśa', 'Space = all this', '#5A3A9C', False),
        ('9. Brahman', 'Brahman = all  ★', '#8B6914', True),
    ]

    ax1.set_title('Mantra Pushpam Ascending Chain\n(Taittirīya Āraṇyaka 1.22)',
                  fontsize=12, color=BLUE, pad=10, fontweight='bold')

    for i, (name, meaning, color, highlighted) in enumerate(chain):
        y = 8.5 - i * 0.9
        # Box
        fc = '#FFF8E8' if highlighted else 'white'
        ec = color if highlighted else '#CCCCCC'
        lw = 2.0 if highlighted else 0.8
        rect = plt.Rectangle((0.3, y - 0.32), 9.2, 0.62,
                               facecolor=fc, edgecolor=ec, linewidth=lw,
                               transform=ax1.transData, zorder=2)
        ax1.add_patch(rect)
        ax1.text(0.7, y, name, fontsize=10, color=color, fontweight='bold',
                 va='center', zorder=3)
        ax1.text(5.2, y, meaning, fontsize=9, color=GREY, va='center',
                 style='italic', zorder=3)

        # Arrow to next
        if i < len(chain) - 1:
            ax1.annotate('', xy=(5, 8.5 - (i+1)*0.9 + 0.32),
                         xytext=(5, y - 0.32),
                         arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=1.2),
                         zorder=1)

    ax1.text(5, -0.3,
             '"He who knows the Year knows Prajāpati;\nHe who knows Prajāpati knows Brahman."',
             ha='center', fontsize=8.5, color=GREY, style='italic')

    # Rung count annotation
    ax1.text(9.8, 8.5, '9 rungs', fontsize=8.5, color='#8B6914',
             rotation=90, va='top', ha='center', style='italic',
             bbox=dict(boxstyle='round', facecolor=LGOLD, edgecolor=GOLD, alpha=0.8))

    # RIGHT: C11 odd numbers
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(-0.5, 10)
    ax2.set_title('Chamakam C11 Anuvāka — Odd Numbers as Samvatsaras\n{1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21}',
                  fontsize=12, color=BLUE, pad=10, fontweight='bold')

    c11_items = [
        (1,  '1 yr',  'Ekam — eka eva prajāpatiḥ (ŚB 11.1.6)', False),
        (3,  '3 yr',  'Trikāla — 3 times / 3 waters (MP)', False),
        (5,  '5 yr',  'Pañcavatsarīya yuga (Vedāṅga Jyotiṣa)', False),
        (7,  '7 yr',  'Sapta prāṇāḥ from Prajāpati (ŚB)', False),
        (9,  '9 yr  ★', '= NUMBER OF MP RUNGS (TA 1.22)', True),
        (11, '11 yr', 'Ekādaśa Rudras; solar cycle ~10.66 yr', False),
        (13, '13 yr', 'Prajāpati = 13th month (ŚB 11.1.1)', False),
        (15, '15 yr', '15 tithis per pakṣa; moon = water (MP)', False),
        (17, '17 yr', 'Prajāpati = 17-fold (ŚB 10.4.2.2)', False),
        (19, '19 yr ★', 'Metonic cycle: 19 yr = 235 lunar months', True),
        (21, '21 yr ★', 'Ekaviṃśaḥ vai prajāpatiḥ (ŚB 10.4.2.2)', True),
    ]

    for i, (n, yr, note, star) in enumerate(c11_items):
        y = 9.0 - i * 0.78
        fc = '#FFF8E8' if star else 'white'
        ec = GOLD if star else '#CCCCCC'
        lw = 2.0 if star else 0.8
        rect = plt.Rectangle((0.2, y - 0.3), 9.4, 0.58,
                               facecolor=fc, edgecolor=ec, linewidth=lw,
                               transform=ax2.transData, zorder=2)
        ax2.add_patch(rect)
        color = GOLD if star else GREY
        ax2.text(0.6, y, yr, fontsize=9.5, color=BLUE if not star else RED,
                 fontweight='bold', va='center', zorder=3)
        ax2.text(2.0, y, note, fontsize=8.5, color=color,
                 va='center', style='italic' if not star else 'normal', zorder=3)

    # Summary annotations
    ax2.text(5, -0.05, 'Sum = 121 = 11²  ·  Product = 21!! = 13.749 Gy',
             ha='center', fontsize=9, color=BLUE, fontweight='bold')
    ax2.text(5, -0.38, '★ = strongest Mantra Pushpam resonances',
             ha='center', fontsize=8.5, color=GOLD, style='italic')

    # Bridge arrow between panels
    fig.patches.append(
        FancyArrowPatch(
            (0.505, 0.42), (0.495, 0.42),
            transform=fig.transFigure,
            arrowstyle='<->', color=RED, linewidth=2,
            mutation_scale=15
        )
    )
    fig.text(0.502, 0.435, 'Samvatsara\n= Prajāpati',
             ha='center', va='bottom', fontsize=8, color=RED,
             style='italic', fontweight='bold')

    fig.suptitle(
        'Figure 4. Structural Correspondence: Mantra Pushpam Time-Chain × Chamakam C11',
        fontsize=13, fontweight='bold', color=BLUE, y=1.01
    )

    plt.tight_layout()
    plt.savefig('/home/claude/fig4_mp_c11.png', bbox_inches='tight')
    plt.close()
    print('Fig 4 saved.')


# ── Figure 5: UNIVERSE AGE (APPENDIX A.3) ────────────────────────────────────
def make_fig5():
    measurements = [
        ('Planck 2018\n(CMB)',              13.787, 0.020, BLUE),
        ('WMAP 9-year',                     13.772, 0.059, '#2E5C8A'),
        ('HST Key Project\n(H₀ = 72)',       13.600, 0.600, GREEN),
        ('Globular cluster\nlower bound',    13.600, 0.800, '#3A7A3A'),
    ]

    N_vedic = 13.7493  # 21!! in Gy

    fig, ax = plt.subplots(figsize=(9, 5))

    # Plot each measurement
    for i, (label, val, sig, color) in enumerate(measurements):
        y = len(measurements) - i
        ax.errorbar(val, y, xerr=sig, fmt='o', color=color,
                    capsize=6, capthick=2, elinewidth=2,
                    markersize=9, markerfacecolor=color,
                    markeredgecolor='white', markeredgewidth=1.5,
                    zorder=3)
        ax.text(val - sig - 0.05, y, label, ha='right', va='center',
                fontsize=9.5, color=color)

        # Sigma annotation
        sigma_dist = abs(N_vedic - val) / sig
        side = '+0.04 Gy' if N_vedic > val else f'−{abs(N_vedic-val):.3f} Gy'
        ax.text(max(measurements, key=lambda x: x[1])[1] + 0.65, y,
                f'{sigma_dist:.2f}σ',
                ha='left', va='center', fontsize=9,
                color='black' if sigma_dist <= 1 else RED)

    # 21!! line
    ax.axvline(N_vedic, color=RED, linewidth=2.5, linestyle='-', zorder=4,
               label=f'21!! = {N_vedic:.4f} Gy')
    ax.text(N_vedic + 0.01, 4.6, '21!! = 13.7493 Gy\n(C11 Chamakam product,\n'
            'samvatsara units)',
            fontsize=9, color=RED, va='top',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF0F0',
                      edgecolor=RED, alpha=0.9))

    # Hubble tension band
    ax.axvspan(12.9, 13.0, alpha=0.08, color=RED, label='H₀=73 estimate')
    ax.axvspan(13.767, 13.807, alpha=0.08, color=BLUE, label='Planck 1σ range')

    ax.set_xlabel('Age of Universe (Gy)', labelpad=8)
    ax.set_yticks([])
    ax.set_xlim(12.6, 14.8)
    ax.set_ylim(0.3, 5.2)
    ax.set_title(
        'Figure 5 (Appendix A.3). 21!! in Samvatsara Units vs. Universe Age Measurements\n'
        'Note: This correspondence does not survive look-elsewhere correction — '
        'see §A.3 for full caveats',
        pad=10, fontsize=11
    )

    # Sigma column header
    ax.text(max(measurements, key=lambda x: x[1])[1] + 0.65, 4.8,
            'σ-distance', ha='left', va='center', fontsize=9.5,
            fontweight='bold', color=GREY)

    legend_elems = [
        Line2D([0], [0], color=RED, linewidth=2.5,
               label=f'21!! = {N_vedic:.4f} Gy (C11 Chamakam product)'),
        mpatches.Patch(facecolor=BLUE, alpha=0.15, label='Planck 2018 ±1σ range'),
        mpatches.Patch(facecolor=RED,  alpha=0.10, label='H₀=73 estimate (Hubble tension)'),
    ]
    ax.legend(handles=legend_elems, loc='lower left', fontsize=9,
              framealpha=0.9, edgecolor=GREY)

    plt.tight_layout()
    plt.savefig('/home/claude/fig5_universe_age.png')
    plt.close()
    print('Fig 5 saved.')


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Generating figures...')
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    print()
    print('All 5 figures saved to /home/claude/')
    print('  fig1_null_distribution.png  — Fig 1 (MC null distribution)')
    print('  fig2_timeline.png           — Fig 2 (Vedic vs Geo timeline)')
    print('  fig3_sigma_bars.png         — Fig 3 (sigma distance bars)')
    print('  fig4_mp_c11.png             — Fig 4 (MP chain + C11 dual)')
    print('  fig5_universe_age.png       — Fig 5 (universe age, Appendix)')
