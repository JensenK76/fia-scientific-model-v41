# Changelog

## v4.2.0

V4.2 is a consolidated runnable release built from the V4.1 null-controlled testbench plus the additional runner and diagnostics used in the N=200/N=400 validation runs.

### Added

- Integrated grid runner in `fia_scientific_v42.py`.
- `--mode self-test` installation smoke test.
- `--preset n400-sigma030` and `--preset n200-sigma030`.
- Sparse spectral acceleration via `scipy.sparse` and `scipy.sparse.linalg`.
- Fast final-spectral mode for large grids.
- `hob_catalog.csv` output.
- `C_metric_quantile_summary.csv` output.
- `C_metric_full_median`, `C_metric_full_q25`, `C_metric_full_q75`, and `C_metric_zero_fraction`.
- Stable output validation in `validation_summary.json`.
- `run_manifest.json`, `timings.csv`, and `failures.csv`.

### Changed

- Main large-grid output naming now uses `run_results.csv`, `timeseries.csv`, `hob_tracks.csv`, `hob_catalog.csv`, `threshold_robustness.csv`, and `paired_differences.csv`.
- Paired-difference columns use explicit names, for example `diff_C_metric_full_full_minus_control`.
- `timeseries.csv` skips expensive spectral metrics on intermediate rows by default in fast mode; final rows and `run_results.csv` contain final spectral endpoints.

### Fixed/hardened

- Empty optional outputs are written as readable CSV files with headers.
- Markdown tables fall back to plain text if `tabulate` is unavailable.
- The package includes a self-test mode for quick local verification.

### Scientific note

V4.2 remains a falsifiable testbench. The strong FIA claim is supported only if the full condition beats equal, beta0, shuffle, and graph-null controls on the preregistered primary endpoint and relevant persistent-hob endpoints with paired uncertainty estimates.
