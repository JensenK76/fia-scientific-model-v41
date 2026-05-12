# FIA Scientific Model V4.2 test report

Test date: 2026-05-11

## Tested files

- `fia_scientific_v42.py`
- `requirements.txt`
- `README.md`

## Commands run

```bash
python -m py_compile fia_scientific_v42.py
python fia_scientific_v42.py --mode self-test --out V42_selftest_output2
python fia_scientific_v42.py --mode grid --out V42_tiny_grid --clean \
  --Ns 80 --sigmas 0.30 --betas 0,1 --Ks 1.2 --phase-noises 0.005 \
  --controls full,equal_identical,beta0_full --reps 2 --steps 30 --burn-in 10 --workers 2
```

## Results

- GitHub package compile: PASS
- GitHub package self-test: PASS, 24/24 conditions, 0 failures
- Compile: PASS
- Self-test: PASS, 24/24 conditions, 0 failures
- Tiny multiprocessing grid: PASS, 12/12 conditions, 0 failures
- `run_results.csv`: readable, no non-finite numeric values
- `timeseries.csv`: readable; intermediate spectral NaN values are expected in fast mode
- `hob_tracks.csv`: readable
- `hob_catalog.csv`: readable
- `threshold_robustness.csv`: readable
- `paired_differences.csv`: readable
- `C_metric_quantile_summary.csv`: readable, no non-finite numeric values

## Note

V4.2 was tested as a code package and smoke/regression runner. The built-in `n400-sigma030` preset matches the previously tested N=400 setup, but the full N=400, reps=20 run is computationally heavier than the smoke tests.
