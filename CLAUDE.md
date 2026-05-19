# CLAUDE.md — DESI_astra / ACM Repository

## What This Repo Is

**ACM (Alternative Clustering Methods)** — an emulator-based cosmological analysis pipeline for galaxy surveys (DESI BGS, EMC). It measures 13+ galaxy clustering statistics, trains neural-network emulators, and performs Bayesian inference on cosmological parameters.

The name `DESI_astra` reflects use of **ASTRA**, a cosmic-web classification algorithm embedded in the pipeline (not the SDSS stellar-parameter pipeline).

---

## What ASTRA Is in This Context

ASTRA labels each galaxy by its **local cosmic environment** (void / sheet / filament / knot):

1. Build a Delaunay triangulation over all points (data galaxies + randoms).
2. For each data point count its triangulation-neighbor data galaxies (`ndata`) and randoms (`nrand`).
3. Compute local density parameter: `r = (ndata - nrand) / (ndata + nrand)`.
4. Classify:

| Type     | r range            |
|----------|--------------------|
| Void     | −1 ≤ r ≤ −0.9     |
| Sheet    | −0.9 < r ≤ 0      |
| Filament |  0 < r ≤ 0.9      |
| Knot     |  0.9 < r ≤ 1      |

**Key FITS output columns:** `TARGETID`, `XCART`, `YCART`, `ZCART`, `RANDITER` (−1 = data, ≥0 = random), `ISDATA`, `NDATA`, `NRAND`, `PVOID`, `PSHEET`, `PFILAMENT`, `PKNOT`, `QUARTILE`.

---

## Key ASTRA-Related Files

| File | Lines | Role |
|------|-------|------|
| `acm/estimators/galaxy_clustering/astra_split.py` | 318 | Core ASTRA implementation |
| `acm/create_dataset_fisher.py` | 400+ | Fisher dataset creation using ASTRA on Abacus mocks |
| `acm/create_dataset_simple_jax.py` | — | JAX-based dataset creation with ASTRA quantile injection |
| `acm/fisher.ipynb` | (5.6 MB) | Fisher information matrix analysis (main notebook) |
| `acm/fisher_matrices.ipynb` | (3.2 MB) | Fisher matrix figures and statistics |

### `astra_split.py` Key Methods

```python
AstraSplit.load_survey_fits()                    # Load original ASTRA FITS survey output
AstraSplit.build_dataframe_from_hod()            # Convert HOD catalog + randoms → ASTRA format
AstraSplit.generate_uniform_randoms_from_hod()   # Sample randoms from box volume
AstraSplit.generate_pairs_classification_probability()  # Delaunay → r values → classify
AstraSplit.classify_type(r)                      # Assign void/sheet/filament/knot
AstraSplit.build_final_classification()          # Quantile-based bucketing (QUARTILE column)
```

---

## Other Estimators (in `acm/estimators/galaxy_clustering/`)

| Estimator file | Method | Output |
|----------------|--------|--------|
| `density_split.py` | Quantile-based clustering | Cross-correlations per quantile |
| `spectrum.py` | Power spectrum P(k,ℓ) | Multipoles ℓ = 0, 2, 4 |
| `jaxmf.py` | Minkowski Functionals (JAX) | Volume, surface, curvature, Euler char |
| `wst.py` | Wavelet Scattering Transform | Scattering coefficients |
| `mst.py` | Minimum Spanning Tree | Correlation dimension |
| `multiplets.py` | Galaxy multiplets | Group sizes |
| `knn.py` | KNN distances | k-th neighbor distance distribution |
| `voxel_voids.py` | Voxel void finder | Void size distribution |
| `pydive.py` | Delaunay-Tessellation voids | Void statistics |
| `polybin3d.py` | Bispectrum | B(k₁, k₂, cos θ) |
| `cic.py` | Counts in Cells | Count distribution |

Computation backends: `jaxpower`, `pypower`, `pyrecon`.

---

## Data Paths

| Data | Path | Format |
|------|------|--------|
| HOD catalogs (z=0.5) | `/pscratch/sd/n/ntbfin/emulator/hods/z0.5/yuan23_prior/c{cosmo:03}_ph{phase:03}/seed{seed}/hod*.fits` | FITS |
| HOD columns | `X_RSD`, `Y_RSD`, `Z_RSD` (redshift-space positions) | — |
| Cosmology params | `/pscratch/sd/e/epaillas/emc/AbacusSummit.csv` | CSV |
| BGS measurements | `/pscratch/sd/s/sbouchar/acm/bgs-{Mr}/measurements/` | `.npy` |
| BGS emulators | `/pscratch/sd/s/sbouchar/acm/bgs-{Mr}/trained_models/` | PyTorch `.ckpt` |
| EMC measurements | `/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/measurements/v1.2/abacus/` | `.npy` |
| EMC emulators | `/global/cfs/cdirs/desicollab/users/epaillas/acm/emc/v1.2/trained_models/best/{stat}/last.ckpt` | PyTorch `.ckpt` |
| ASTRA Fisher datasets | `dataset_covariance_astra_*.npz`, `dataset_derivatives_astra_*.npz` | `.npz` |

---

## Core Pipeline Workflow

```
1. HOD catalogs (Abacus + Yuan+23 prior)
         ↓
2. ASTRA classification (optional)  ← astra_split.py
   → void / sheet / filament / knot labels
         ↓
3. Measure clustering statistics    ← scripts/emc/measurements/abacus_base.py
   → 13+ stats (spectrum, TPCF, Minkowski, WST, ...)
         ↓
4. Compress to xarray .npy          ← scripts/emc/measurements/compress_files.py
   → coords: (cosmo_idx, hod_idx, phase_idx) × bins
         ↓
5. Train emulator (sunbird/PyTorch)  ← scripts/emc/training/train_sunbird.py
   → (Ωm, Ωb, h, ns, σ8, HOD params) → statistics
         ↓
6. Fisher analysis                   ← scripts/emc/fisher/greedy_fisher.py
   → F_ij = dμ/dθᵢᵀ · C⁻¹ · dμ/dθⱼ  (JAX jacrev)
         ↓
7. Inference (PocoMC / MCMC)         ← scripts/emc/inference/inference_abacus.py
```

---

## Fisher Information Details

- **Parameter derivatives**: computed with JAX `jacrev()` on the emulator.
- **Covariance correction**: Percival-Fisher correction `f(n_samples, n_data, n_params)`.
- **Greedy bin selection**: iteratively add bins that maximize `log det(F)`.
- **ASTRA interface**: cosmic-web type probabilities can be used as weights or split the Fisher matrix.
- **Notebooks**: `acm/fisher.ipynb` (main), `acm/fisher_matrices.ipynb` (figures).

---

## Example Notebooks (`nb/`)

| Notebook | Purpose |
|----------|---------|
| `density_split_examples.ipynb` | Density split tutorial with HOD mocks |
| `minkowski_examples.ipynb` | JAX Minkowski Functionals demo |
| `multiplets_examples.ipynb` | Galaxy multiplet detection |
| `observables_example.ipynb` | xarray Observable class usage |
| `wst_examples.ipynb` | Wavelet Scattering Transform |
| `power_spectrum_examples.ipynb` | P(k) multipoles with jaxpower |
| `posterior_predictive_checks.ipynb` | PPC / model validation |
| `check_cosmo.ipynb` | Environment verification |

---

## Projects in Scope

- **EMC** — Extended Cosmological Mocks: full clustering on 2000 Mpc/h Abacus boxes.
- **BGS** — Bright Galaxy Survey: luminosity-selected (Mr) analysis with DESI geometry.
- **DESI** — Survey-specific: fiber assignment, lightcone, cut-sky, cubic mocks.
- **EFT** — Effective Field Theory inference (`scripts/eft/`).

---

## Configuration Files

| File | Content |
|------|---------|
| `acm/utils/paths/projects.yaml` | Project registry: paths, model checkpoints, HOD dirs |
| `acm/utils/paths/Abacus.yaml` | Abacus cosmology IDs and simulation parameters |
| `acm/hod/box.yaml` | HOD parameter defaults for cubic boxes |
| `acm/hod/lightcone.yaml` | HOD parameter defaults for lightcones |
| `scripts/bgs/inference/config_*.yaml` | BGS emulator paths and inference settings |

---

## Running Scripts Interactively at NERSC

### Python environment

Always use the **cosmodesi** environment (maintained by A. Dematti); it ships
pycorr, fitsio, scipy, pandas, and the full LSS stack:

```bash
unset PYTHONPATH   # clear any ~/.bashrc override first
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
```

Available tags: `main` (rolling), `2026_02`, `2025_12`, `2025_05`, `dr1`.

### Interactive allocation

```bash
salloc -N 1 -C cpu -q interactive -t 30:00 -A desi -c 8 --mem=32G
```

Increase `-t` if the job needs more than 30 min (Delaunay on large catalogs
can take 5–10 min for ~125k points).

### Full sequence

```bash
# 1. From the login node — request a node
salloc -N 1 -C cpu -q interactive -t 30:00 -A desi -c 8 --mem=32G

# 2. On the compute node — load environment
unset PYTHONPATH
source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main

# 3. Run the script
cd /pscratch/sd/f/forero/DESI_astra
srun -n 1 -c 8 python scripts/astra_box_basic.py
```

### Basic ASTRA box script

`scripts/astra_box_basic.py` — end-to-end demo pipeline:

1. **Load** `c000_ph000/seed0/hod000.fits` from the EMC HOD catalog
2. **Apply RSD** (los=z): positions = `X_PERP`/`Y_PERP`/`Z_RSD`, AP-corrected by `Q_PAR`/`Q_PERP` from the FITS header
3. **Cut** a 500 Mpc/h cube centred at the origin (~62k galaxies)
4. **Generate** uniform randoms (`N_RAND=1×` the data count)
5. **Run ASTRA**: Delaunay triangulation → local density `r = (ndata−nrand)/(ndata+nrand)` for every point (data and randoms alike)
6. **Split into quantiles**: bin edges derived from the data `r` distribution via `pd.qcut`; same edges applied to randoms via `pd.cut` — both populations split at identical density thresholds
7. **Save** per-quantile files:
   - `data_quantile_q{1..4}.npy` — galaxy positions
   - `rand_quantile_q{1..4}.npy` — random positions (carry real density information, not uniform)
8. **Generate geometry randoms** (`N_RAND_GEOM=5×` data, uniform in the subbox, separate from ASTRA randoms) — required because the subbox has **open boundaries, not periodic BC**
9. **Compute 2PCF** (monopole ℓ=0 + quadrupole ℓ=2) for each data and random quantile using the **Landy-Szalay estimator** `(DD − 2DR + RR) / RR` with the geometry randoms; no `boxsize` argument passed to pycorr
10. **Save** `multipoles_tpcf_data_q{q}.npz` and `multipoles_tpcf_rand_q{q}.npz` (keys: `s`, `xi0`, `xi2`)

> **Note on randoms**: Two separate random catalogs are used.  The *ASTRA randoms* (`N_RAND=1×`) enter the Delaunay triangulation and get their own density label — their split into quantiles is physically meaningful.  The *geometry randoms* (`N_RAND_GEOM=5×`, generated with `SEED+1`) are uniform and used only to correct for the open-boundary geometry in the LS estimator.

Output directory: `/pscratch/sd/f/forero/sims/test/astra_basic/`

`scripts/plot_astra_basic.py` — produces 5 figures in `…/plots/`:

| File | Content |
|------|---------|
| `data_monopole_per_quantile.png` | s²ξ₀(s) for data Q1–Q4 |
| `data_quadrupole_per_quantile.png` | s²ξ₂(s) for data Q1–Q4 |
| `data_multipoles_all_quantiles.png` | Monopole + quadrupole side by side |
| `rand_monopole_per_quantile.png` | s²ξ₀(s) for random Q1–Q4 (non-zero because randoms trace the cosmic web) |
| `rand_quadrupole_per_quantile.png` | s²ξ₂(s) for random Q1–Q4 |
| `data_vs_rand_monopole.png` | Data vs randoms monopole, one panel per quantile |
