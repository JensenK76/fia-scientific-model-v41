[README (1).md](https://github.com/user-attachments/files/27565647/README.1.md)
# fia-scientific-model-v41
Null-controlled Python testbench for FIA-inspired mass-diverse relational networks, testing structural cohesion, stiffness-weighted metric gaps, persistent hobs, factorial diversity channels, and strong null controls.
# FIA Scientific Model V4.1

**Null-controlled hob-cohesion testbench for FIA-inspired mass-diverse relational networks.**

This repository contains a Python implementation of **FIA Scientific Model V4.1**, a computational testbench for studying whether mass-diverse relational networks can generate persistent, phase-coherent, spectrally cohesive candidate hobs under strong null controls.

The model is deliberately conservative: it is designed not merely to produce visually interesting structures, but to test whether the full FIA-inspired model outperforms relevant controls. A strong FIA-hob claim is supported only if the full model beats null models on preregistered cohesion and persistence endpoints.

## What this model tests

The model studies FIA-inspired hobs of the form

\[
H=(V,m,A),
\]

where:

- \(V\) is a finite set of units,
- \(m_i>0\) are primitive mass/capacity weights,
- \(G^{(0)}\) is a baseline relational graph,
- \(A\) is a mass-mismatch-suppressed relation matrix.

The mismatch between two units is

\[
\Delta_{ij}=\left|\log\frac{m_i}{m_j}\right|,
\]

and the relation weights are

\[
A_{ij}=G^{(0)}_{ij}\exp(-\beta\Delta_{ij}).
\]

The capacity-weighted phase dynamics is

\[
\dot\theta_i
=
\omega_i
+
\frac{K}{m_i d_{\mathrm{ref}}}
\sum_j A_{ij}\sin(\theta_j-\theta_i)
+
\sqrt{dt}\,\xi_i(t).
\]

The three diversity channels are tested separately and jointly:

1. **Kernel channel**: mass mismatch affects relations through \(A_{ij}\).
2. **Capacity channel**: masses enter the response through \(m_i^{-1}\).
3. **Omega channel**: masses can affect natural frequencies \(\omega_i\).

## What this model does *not* claim

This code does **not** prove FIA as a physical theory.

It does **not** derive electromagnetism, gauge symmetry, photons, quantum electrodynamics, or the Standard Model.

It tests a narrower pre-Maxwell question:

> Can FIA-inspired mass-diverse relational networks produce persistent, null-controlled, spectrally cohesive candidate hobs?

Current V4.1 tests support a limited conclusion: **mass-mismatch-suppressed relations robustly fragment structural and dynamic cohesion**. They do **not yet** establish robust unique FIA-hob emergence beyond all controls.

## Score versus metric

A central methodological point is the separation between detection score and physical cohesion metric.

Candidate hobs are detected using a dimensionless score:

\[
W^{score}_{ij}
=
\operatorname{clip}\left(\frac{A_{ij}}{A_{\mathrm{ref}}},0,1\right)
\frac{1+\cos(\theta_i-\theta_j)}{2}.
\]

Physical/stiffness cohesion is measured using:

\[
W^{metric}_{ij}
=
\frac{K}{d_{\mathrm{ref}}}
A_{ij}\max(\cos(\theta_i-\theta_j),0).
\]

The key distinction is:

> A normalized resonance score is not the same as physical cohesion.

## Primary endpoint

The preregistered primary endpoint is

\[
C_{metric}^{full}=\lambda_2(L_{W^{metric}},M),
\]

where \(M=\operatorname{diag}(m_i)\).

The main paired contrast is

\[
\Delta C_{metric}^{control}
=
C_{metric}^{full}-C_{metric}^{control}.
\]

A strong FIA-hob claim requires the full model to outperform relevant controls.

## Secondary endpoints

The script also reports:

- \(C_A=\lambda_2(L_A,M)\), structural relation cohesion before phase weighting,
- \(C_{metric}/C_A\), dynamic realization of structural cohesion,
- persistent hob count,
- maximum persistent hob size,
- persistent hob lifetime,
- threshold robustness,
- negative stiffness fraction,
- signed Laplacian minimum eigenvalue,
- factorial diversity-channel effects,
- paired bootstrap confidence intervals,
- paired sign-flip tests.

## Null controls

V4.1 includes strong null and factorial controls:

| Control | Purpose |
|---|---|
| `equal_identical` | No mass diversity and no hidden omega diversity. |
| `beta0_full` | Mass diversity present, but mismatch kernel disabled. |
| `kernel_only` | Only mass-dependent relation kernel active. |
| `capacity_only` | Only capacity channel active. |
| `omega_only` | Only mass-dependent frequency channel active. |
| `kernel_capacity` | Kernel + capacity active. |
| `kernel_omega` | Kernel + omega active. |
| `capacity_omega` | Capacity + omega active. |
| `full` | Kernel + capacity + omega active. |
| `shuffle_kernel_fixed` | Tests whether precise mass-relation placement matters. |
| `shuffle_capacity_fixed` | Tests whether precise capacity placement matters. |
| `shuffle_omega_fixed` | Tests whether precise omega placement matters. |
| `static_phase` | Tests false hob detection in non-dynamic random phases. |
| `random_graph_full` | Tests graph-topology dependence. |
| `degree_preserving_rewire_full` | Degree-preserving graph null. |

## Installation

Recommended Python version: **Python 3.10+**.

Install dependencies:

```bash
pip install numpy pandas matplotlib
```

`matplotlib` is optional if you run with `--no-plots`.

## Quick start

Run a small smoke test:

```bash
python fia_scientific_v41.py --quick --no-plots
```

Run the targeted V4.1 control test:

```bash
python fia_scientific_v41.py --targeted --no-plots
```

Run a medium test with plots:

```bash
python fia_scientific_v41.py --medium
```

Custom run:

```bash
python fia_scientific_v41.py --out FIA_V41_output --N 120 --steps 180 --reps 12 --no-plots
```

Available options include:

```bash
--quick
--targeted
--medium
--out <directory>
--no-plots
--reps <int>
--N <int>
--steps <int>
--normalization global|local_degree
--phase-noise <float>
--bootstrap <int>
```

## Output files

The script writes a complete output directory. Important files include:

| File | Meaning |
|---|---|
| `v41_run_results.csv` | Final per-condition results. |
| `v41_timeseries.csv` | Time-resolved diagnostics. |
| `v41_hob_tracks.csv` | Persistent hob tracks. |
| `v41_summary.csv` | Grouped means and standard errors. |
| `v41_paired_differences.csv` | Per-seed full-minus-control contrasts. |
| `v41_paired_difference_summary.csv` | Bootstrap CIs and paired sign-flip summaries. |
| `v41_factorial_effects.csv` | Per-seed factorial channel effects. |
| `v41_factorial_effect_summary.csv` | Summary of kernel/capacity/omega effects. |
| `v41_threshold_robustness.csv` | Full threshold-grid diagnostics. |
| `v41_threshold_robustness_endpoint.csv` | Condensed threshold-robustness endpoint. |
| `v41_theory_checks.csv` | Internal mathematical sanity checks. |
| `v41_decision_report.md` | Automatic scientific decision report. |
| `v41_config.json` | Run configuration. |

If plots are enabled, figures are written to:

```text
<output_dir>/figures/
```

## Interpreting results

The most important question is not whether `full` produces hobs.

The important question is:

> Does `full` outperform the relevant controls on paired endpoints?

In particular, inspect:

```text
v41_decision_report.md
v41_paired_difference_summary.csv
v41_factorial_effect_summary.csv
v41_threshold_robustness_endpoint_summary.csv
```

A conservative interpretation rule is:

- If `full` beats controls on `C_metric_full` and persistent hob endpoints, FIA-hob evidence is strengthened.
- If `full` only differs from equal/beta-zero controls but not from kernel/shuffle controls, the result supports mismatch fragmentation but not unique FIA-hob emergence.
- If `static_phase` produces persistent hobs, the hob detector is too permissive.
- If graph nulls reproduce the result, topology rather than FIA mass structure may be driving the effect.

## Repository structure

Suggested layout:

```text
fia-scientific-model-v41/
├── fia_scientific_v41.py
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
```

Generated output folders should normally not be committed unless you deliberately want to publish a small reproducible example.

## Recommended article use

This model is best presented as a pre-Maxwell hob-cohesion testbench. It supports a cautious claim such as:

> V4.1 provides a null-controlled framework for testing whether FIA-inspired mass-diverse networks generate persistent, spectrally cohesive candidate hobs. Current tests support mass-mismatch fragmentation but do not yet establish unique FIA-hob emergence beyond all controls.

## Citation

If this repository supports a manuscript or preprint, cite the corresponding manuscript and include the software version or commit hash.

Suggested software citation format:

```text
Jensen, K. H. FIA Scientific Model V4.1: Null-Controlled Hob-Cohesion Testbench for FIA-Inspired Mass-Diverse Relational Networks. Python software, 2026.
```

## License

This repository is released under the MIT License unless stated otherwise. See `LICENSE`.
