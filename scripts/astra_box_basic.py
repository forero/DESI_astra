#!/usr/bin/env python3
"""
Basic ASTRA pipeline on a 500 Mpc/h subbox.

Steps
-----
1. Load HOD mock (los=z, RSD applied via X_PERP / Y_PERP / Z_RSD columns)
2. Cut a 500 Mpc/h cube centred at the origin
3. Generate uniform randoms for ASTRA
4. Run ASTRA (Delaunay triangulation → local density r → quantile labels)
5. Split both data and randoms into quantiles using the same r bin edges;
   save per-quantile position files
6. Generate a separate set of uniform geometry randoms (5× data) for the
   Landy-Szalay estimator — the subbox has open boundaries, NOT periodic BC
7. Compute the 2PCF (monopole ℓ=0, quadrupole ℓ=2) for each data quantile
   and each random quantile using the Landy-Szalay estimator with pycorr

Usage (from inside an interactive allocation)
---------------------------------------------
  salloc -N 1 -C cpu -q interactive -t 2:00:00 -A desi
  srun -n 1 -c 32 python scripts/astra_box_basic.py
"""

import sys
import numpy as np
import pandas as pd
import fitsio
from pathlib import Path

# --- make sure the repo is importable -----------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from acm.estimators.galaxy_clustering.astra_split import AstraSplit
from pycorr import TwoPointCorrelationFunction

# ── configuration ──────────────────────────────────────────────────────────────
HOD_FILE = Path(
    '/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior'
    '/c000_ph000/seed0/hod000.fits'
)
OUT_DIR  = Path('/pscratch/sd/f/forero/sims/test/astra_basic')
BOXSIZE  = 500.0    # Mpc/h — side length of the subbox to cut
LOS      = 'z'     # line-of-sight axis
N_Q           = 4   # number of ASTRA quantiles
N_RAND        = 1   # ASTRA randoms = N_RAND × n_data (used for Delaunay classification)
N_RAND_GEOM   = 5   # geometry randoms = N_RAND_GEOM × n_data (used for Landy-Szalay 2PCF)
SEED          = 42
NTHREADS      = 8

# 2PCF s-µ binning
S_EDGES  = np.linspace(0, 150, 31)   # 30 bins, 0–150 Mpc/h
MU_EDGES = np.linspace(-1, 1, 241)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load HOD and apply RSD ──────────────────────────────────────────────────
print('Loading HOD catalog ...')
data, hdr = fitsio.read(str(HOD_FILE), header=True)

q_par  = hdr['Q_PAR']    # Alcock-Paczynski dilation along LOS
q_perp = hdr['Q_PERP']   # Alcock-Paczynski dilation transverse

# For los=z: transverse coords use X_PERP / Y_PERP, RSD along z uses Z_RSD
pos_full = np.c_[
    data['X_PERP'] / q_perp,
    data['Y_PERP'] / q_perp,
    data['Z_RSD']  / q_par,
]
print(f'  Full box: {len(pos_full):,} galaxies')

# ── 2. Cut 500 Mpc/h subbox centred at the origin ─────────────────────────────
half = BOXSIZE / 2
mask = (
    (pos_full[:, 0] >= -half) & (pos_full[:, 0] < half) &
    (pos_full[:, 1] >= -half) & (pos_full[:, 1] < half) &
    (pos_full[:, 2] >= -half) & (pos_full[:, 2] < half)
)
positions = pos_full[mask].astype(np.float64)
boxsize   = np.array([BOXSIZE, BOXSIZE, BOXSIZE])
print(f'  Subbox ({BOXSIZE:.0f} Mpc/h): {len(positions):,} galaxies')

# ── 3. Generate uniform randoms for ASTRA ─────────────────────────────────────
print('\nGenerating ASTRA randoms ...')
astra = AstraSplit()
rand_positions = astra.generate_uniform_randoms_from_hod(
    positions=positions,
    boxsize=boxsize,
    n_factor=N_RAND,
    seed=SEED,
)
print(f'  {len(rand_positions):,} randoms generated  (factor={N_RAND}×)')

# ── 4. Run ASTRA ───────────────────────────────────────────────────────────────
print('\nBuilding ASTRA dataframe (data + randoms) ...')
df_full = astra.build_dataframe_from_hod(
    positions=positions,
    random_positions=rand_positions,
)

print('Running ASTRA (Delaunay triangulation + density classification) ...')
_, class_rows, _ = astra.generate_pairs_classification_probability(df_full)

print(f'Assigning {N_Q} quantile labels ...')
df_class = astra.build_final_classification(class_rows, n_quantiles=N_Q)

# ── 5. Split data AND randoms into quantiles using the same r bin edges ────────
# build_final_classification assigns QUARTILE only to data.  We recompute the
# same bin edges and apply them to randoms so both populations are split at the
# same density thresholds.
print('\nAssigning quantile labels to randoms ...')

n_data = len(positions)

data_mask = df_class['ISDATA_BOOL'] & df_class['r'].notna()
rand_mask  = ~df_class['ISDATA_BOOL'] & df_class['r'].notna()

# Bin edges derived from the data r distribution (same as build_final_classification)
_, bin_edges = pd.qcut(df_class.loc[data_mask, 'r'], N_Q,
                        retbins=True, duplicates='drop')
bin_edges[0]  = -np.inf   # ensure all randoms fall inside a bin
bin_edges[-1] =  np.inf

df_class.loc[rand_mask, 'QUARTILE'] = pd.cut(
    df_class.loc[rand_mask, 'r'],
    bins=bin_edges,
    labels=list(range(1, N_Q + 1)),
    include_lowest=True,
).astype(float)

print('\nSaving per-quantile catalogs ...')
for q in range(1, N_Q + 1):
    # data galaxies in quantile q  (TARGETID directly indexes positions)
    df_q_data = df_class[df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
    pos_q = positions[df_q_data['TARGETID'].values]
    np.save(OUT_DIR / f'data_quantile_q{q}.npy', pos_q)
    print(f'  Q{q} data:    {len(pos_q):,} galaxies')

    # randoms in quantile q  (TARGETID - n_data indexes rand_positions)
    df_q_rand = df_class[~df_class['ISDATA_BOOL'] & (df_class['QUARTILE'] == q)]
    rand_q = rand_positions[df_q_rand['TARGETID'].values - n_data]
    np.save(OUT_DIR / f'rand_quantile_q{q}.npy', rand_q)
    print(f'  Q{q} randoms: {len(rand_q):,} randoms')

# ── 6. Generate geometry randoms for Landy-Szalay estimator ───────────────────
# The subbox has open boundaries (not periodic), so we need explicit randoms
# for the LS estimator (DD - 2DR + RR) / RR.  These are separate from the
# ASTRA randoms used in the Delaunay triangulation.
print('\nGenerating geometry randoms for 2PCF ...')
rng_geom = np.random.default_rng(SEED + 1)
geom_randoms = rng_geom.uniform(
    low  = -boxsize / 2,
    high =  boxsize / 2,
    size = (N_RAND_GEOM * len(positions), 3),
)
print(f'  {len(geom_randoms):,} geometry randoms  (factor={N_RAND_GEOM}×)')

# ── 7. Compute 2PCF (monopole + quadrupole) ───────────────────────────────────
edges = (S_EDGES, MU_EDGES)

def compute_and_save_tpcf(positions_in, label, out_stem):
    print(f'  {label} ({len(positions_in):,}) ...')
    xi = TwoPointCorrelationFunction(
        'smu',
        edges=edges,
        data_positions1=positions_in,
        randoms_positions1=geom_randoms,   # Landy-Szalay; no boxsize (non-periodic)
        engine='corrfunc',
        nthreads=NTHREADS,
        compute_sepsavg=True,
        position_type='pos',
        los=LOS,
    )
    xi.save(str(OUT_DIR / f'{out_stem}.npy'))
    s, multipoles = xi(ells=(0, 2), return_sep=True)
    xi0, xi2 = multipoles[0], multipoles[1]
    np.savez(OUT_DIR / f'multipoles_{out_stem}.npz', s=s, xi0=xi0, xi2=xi2)
    print(f'    xi0[15] = {xi0[15]:.4f}  xi2[15] = {xi2[15]:.4f}')

print('\nComputing 2PCF for data quantiles ...')
for q in range(1, N_Q + 1):
    pos_q = np.load(OUT_DIR / f'data_quantile_q{q}.npy')
    compute_and_save_tpcf(pos_q, f'data Q{q}', f'tpcf_data_q{q}')

print('\nComputing 2PCF for random quantiles ...')
for q in range(1, N_Q + 1):
    rand_q = np.load(OUT_DIR / f'rand_quantile_q{q}.npy')
    compute_and_save_tpcf(rand_q, f'randoms Q{q}', f'tpcf_rand_q{q}')

print('\n=== Done ===')
print(f'Output files in {OUT_DIR}:')
for fn in sorted(OUT_DIR.iterdir()):
    print(f'  {fn.name}')
