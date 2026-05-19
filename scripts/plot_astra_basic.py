#!/usr/bin/env python3
"""
Basic plots from astra_box_basic.py output.

Run from any node (no srun needed — CPU-only, lightweight):
  python scripts/plot_astra_basic.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR  = Path('/pscratch/sd/f/forero/sims/test/astra_basic')
PLOT_DIR = OUT_DIR / 'plots'
PLOT_DIR.mkdir(exist_ok=True)

N_Q     = 4
COLORS  = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8']  # Q1=voids→Q4=knots
LABELS  = [f'Q{q}' for q in range(1, N_Q + 1)]

# ── load ──────────────────────────────────────────────────────────────────────
data_quants = []
rand_quants = []
for q in range(1, N_Q + 1):
    d = np.load(OUT_DIR / f'multipoles_tpcf_data_q{q}.npz')
    data_quants.append({'s': d['s'], 'xi0': d['xi0'], 'xi2': d['xi2']})
    r = np.load(OUT_DIR / f'multipoles_tpcf_rand_q{q}.npz')
    rand_quants.append({'s': r['s'], 'xi0': r['xi0'], 'xi2': r['xi2']})


# ── Figure 1: data monopole per quantile ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for d, label, color in zip(data_quants, LABELS, COLORS):
    ax.plot(d['s'], d['s']**2 * d['xi0'], color=color, lw=2, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_0(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Data monopole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'data_monopole_per_quantile.png', dpi=150)
print(f'Saved {PLOT_DIR}/data_monopole_per_quantile.png')

# ── Figure 2: data quadrupole per quantile ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for d, label, color in zip(data_quants, LABELS, COLORS):
    ax.plot(d['s'], d['s']**2 * d['xi2'], color=color, lw=2, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_2(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Data quadrupole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'data_quadrupole_per_quantile.png', dpi=150)
print(f'Saved {PLOT_DIR}/data_quadrupole_per_quantile.png')

# ── Figure 3: data monopole + quadrupole side by side ────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for d, label, color in zip(data_quants, LABELS, COLORS):
    axes[0].plot(d['s'], d['s']**2 * d['xi0'], color=color, lw=2, label=label)
    axes[1].plot(d['s'], d['s']**2 * d['xi2'], color=color, lw=2, label=label)
for ax, title, ell in zip(axes, ['Monopole', 'Quadrupole'], [0, 2]):
    ax.axhline(0, color='k', lw=0.8, ls='--')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(rf'$s^2\,\xi_{ell}(s)\ [h^{{-2}}\,\mathrm{{Mpc}}^2]$')
    ax.set_title(f'Data {title}')
    ax.set_xlim(0, 150)
    ax.legend(fontsize=8)
fig.suptitle('ASTRA data quantile 2PCF  —  500 Mpc/h box, los=z', y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'data_multipoles_all_quantiles.png', dpi=150, bbox_inches='tight')
print(f'Saved {PLOT_DIR}/data_multipoles_all_quantiles.png')

# ── Figure 4: random monopole per quantile (sanity check) ────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for r, label, color in zip(rand_quants, LABELS, COLORS):
    ax.plot(r['s'], r['s']**2 * r['xi0'], color=color, lw=2, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_0(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Random monopole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'rand_monopole_per_quantile.png', dpi=150)
print(f'Saved {PLOT_DIR}/rand_monopole_per_quantile.png')

# ── Figure 5: random quadrupole per quantile ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for r, label, color in zip(rand_quants, LABELS, COLORS):
    ax.plot(r['s'], r['s']**2 * r['xi2'], color=color, lw=2, label=label)
ax.axhline(0, color='k', lw=0.8, ls='--')
ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
ax.set_ylabel(r'$s^2\,\xi_2(s)\ [h^{-2}\,\mathrm{Mpc}^2]$')
ax.set_title('Random quadrupole per ASTRA quantile')
ax.legend(title='Q1=underdense → Q4=overdense')
ax.set_xlim(0, 150)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'rand_quadrupole_per_quantile.png', dpi=150)
print(f'Saved {PLOT_DIR}/rand_quadrupole_per_quantile.png')

# ── Figure 7: data vs randoms monopole per quantile ──────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
for idx, (q, ax) in enumerate(zip(range(1, N_Q + 1), axes.flat)):
    d, r, color = data_quants[idx], rand_quants[idx], COLORS[idx]
    s = d['s']
    ax.plot(s, s**2 * d['xi0'], color=color, lw=2, label='data')
    ax.plot(s, s**2 * r['xi0'], color=color, lw=1.5, ls='--', label='randoms')
    ax.axhline(0, color='k', lw=0.8, ls=':')
    ax.set_title(f'Q{q}')
    ax.set_xlabel(r'$s\ [h^{-1}\,\mathrm{Mpc}]$')
    ax.set_ylabel(r'$s^2\,\xi_0$')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 150)
fig.suptitle('Data vs ASTRA-random monopole per quantile', y=1.01)
fig.tight_layout()
fig.savefig(PLOT_DIR / 'data_vs_rand_monopole.png', dpi=150, bbox_inches='tight')
print(f'Saved {PLOT_DIR}/data_vs_rand_monopole.png')

plt.close('all')
print('Done.')
