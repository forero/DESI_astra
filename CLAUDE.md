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
