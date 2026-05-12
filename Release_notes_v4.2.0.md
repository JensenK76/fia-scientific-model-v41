# v4.2.0 — Sparse Grid Runner + Quantile Diagnostics

This release consolidates the FIA V4.1 model and the large-grid runner into a single V4.2 script.

## Highlights

- Single-file executable: `fia_scientific_v42.py`
- Sparse spectral acceleration for N=200+ runs
- Built-in `--mode self-test`
- Built-in N=400 preset:

```bash
python fia_scientific_v42.py --preset n400-sigma030 --out V42_N400_sigma030 --clean --workers 4
```

- New output: `hob_catalog.csv`
- New output: `C_metric_quantile_summary.csv`
- New diagnostic: `C_metric_zero_fraction`

## Recommended install

```bash
python -m pip install -r requirements.txt
```

## Expected output files

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

## Interpretation note

V4.2 should be read as a null-controlled scientific testbench. It does not by itself prove the FIA hypothesis. Full-model results must be interpreted through paired controls, especially `equal_identical`, `beta0_full`, `shuffle_kernel_fixed`, and graph-null controls.
