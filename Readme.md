# FIA Scientific Model V4.2

FIA Scientific Model V4.2 is a null-controlled testbench for FIA-inspired mass-diverse relational networks. It extends the V4.1 model with a consolidated grid runner, sparse spectral acceleration, hob catalog output, paired control contrasts, and quantile diagnostics for the primary metric endpoint.

The preregistered primary endpoint is:

```text
C_metric_full = lambda_2(L_W_metric, M_capacity)
W_metric_ij = (K / d_ref) A_ij max(cos(theta_i - theta_j), 0)
```

Secondary endpoints include persistent hob counts, persistent hob sizes, structural gap `C_A_full`, dynamic realization `C_metric_over_C_A_full`, threshold robustness, signed-stiffness diagnostics, and `C_metric_zero_fraction`.

## Install

```bash
python -m pip install -r requirements.txt
```

`scipy` is strongly recommended for N=200+ runs. Without scipy, the code falls back to dense eigensolves, which are much slower for large grids.

## Self-test

```bash
python fia_scientific_v42.py --mode self-test --out V42_selftest
```

Expected core output files:

```text
run_results.csv
timeseries.csv
hob_tracks.csv
hob_catalog.csv
threshold_robustness.csv
paired_differences.csv
summary_by_condition.csv
paired_difference_summary.csv
C_metric_quantile_summary.csv
run_report.md
validation_summary.json
run_manifest.json
timings.csv
failures.csv
```

## N=400 sigma=0.30 preset

This preset matches the tested N=400 setup:

```text
N = 400
sigma_m = 0.30
beta = 0, 1, 2, 4
K = 1.2, 1.6
phase_noise = 0.005
steps = 500
burn_in = 100
reps = 20
controls = full, equal_identical, beta0_full, kernel_only, kernel_capacity,
           shuffle_kernel_fixed, static_phase, degree_preserving_rewire_full
```

Run:

```bash
python fia_scientific_v42.py \
  --preset n400-sigma030 \
  --out V42_N400_sigma030 \
  --clean \
  --workers 4
```

## Custom grid example

```bash
python fia_scientific_v42.py \
  --mode grid \
  --out V42_custom \
  --clean \
  --Ns 200,400 \
  --sigmas 0.30 \
  --betas 0,1,2,4 \
  --Ks 1.2,1.6 \
  --phase-noises 0.005 \
  --steps 500 \
  --burn-in 100 \
  --reps 20 \
  --controls full,equal_identical,beta0_full,kernel_only,kernel_capacity,shuffle_kernel_fixed,static_phase,degree_preserving_rewire_full \
  --workers 4
```

## Standard/legacy mode

V4.2 also keeps the older single-config experiment interface:

```bash
python fia_scientific_v42.py --mode standard --quick --no-plots
```

## Important output notes

`run_results.csv` contains final-snapshot endpoints for every completed condition.

`timeseries.csv` uses fast mode by default: intermediate rows include R/hob/edge diagnostics, while expensive full-graph spectral metrics are computed on final rows only. Use `--full-timeseries-spectral` to compute spectral metrics at every observation point. This is much slower.

`hob_catalog.csv` is a catalog version of `hob_tracks.csv` with `catalog_type` set to `persistent_track` or `transient_track`.

`C_metric_quantile_summary.csv` contains:

```text
C_metric_full_mean
C_metric_full_std
C_metric_full_median
C_metric_full_q25
C_metric_full_q75
C_metric_zero_fraction
```

`C_metric_zero_fraction` is defined as the fraction of reps where:

```python
np.isclose(C_metric_full, 0.0, atol=1e-12, rtol=0.0)
```

## Controls included in the main V4.2 preset

```text
full
equal_identical
beta0_full
kernel_only
kernel_capacity
shuffle_kernel_fixed
static_phase
degree_preserving_rewire_full
```

## Scientific interpretation

V4.2 should be read as a falsifiable testbench, not as proof of FIA. A strong FIA claim is supported only when the full condition beats equal, beta0, shuffle, and graph-null controls on the preregistered primary endpoint and persistent-hob endpoints with paired uncertainty estimates. If those contrasts are inconclusive or negative, the supported conclusion is limited to mismatch-driven structural/dynamic fragmentation.
