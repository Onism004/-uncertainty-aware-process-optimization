# Uncertainty-Aware Machine Learning for Industrial Process Optimization

Independent research (2026) — advised by Prof. Lu Lu, University of Jinan.
Author: Mingshan Li (Guangdong Technion – Israel Institute of Technology)

## Motivation

In data-sparse industrial optimization, a surrogate model's point prediction can be confidently wrong: a naive score-maximizing optimizer chases the model's hallucination into regions the data never covered. This 12-week study builds and validates an uncertainty-aware alternative on an aluminum silicate fiber production quality model (simulation-based prototype).

## Key results

| Operating point | Predicted score μ | Epistemic σf | True score |
|---|---|---|---|
| Current process setting | 85.9 | 3.9 | 86.6 |
| Naive optimum (λ = 0) | 109.9 (> full mark — hallucination) | 7.25 | 83.0 (below baseline) |
| Robust optimum (λ = 10) | 100.2 | 4.21 (−42%) | 98.5 |

- **Robustness across seeds**: true score 97.5 ± 0.9 (λ = 10) vs 88.4 ± 6.6 (λ = 0), a ~7× variance collapse; hallucinated optima appeared in 3 of 5 naive runs.
- **Surrogate quality**: Gaussian-process regression, 5-fold CV RMSE 3.97, R² ≈ 0.70; the 95% predictive band shows 100% empirical coverage; observation-noise estimate 3.91 (true 4.0).
- **Decision rule**: under an epistemic-uncertainty budget σf ≤ 4.5, λ* = 7 recovers 98.2 true quality — "when to trust the optimizer" becomes a checkable criterion.
- **Active-learning pre-study**: adding samples in already-covered regions does not cure corner hallucinations (corner σf still 5.7 after 20 added points) — sampling must follow the σf map.

## Method notes

- The λ-objective penalizes **epistemic** uncertainty σf only; aleatoric noise is constant in the decision variables and would act as a constant shift.
- GA operators (elitist top-1/3 selection, convex-combination crossover, 10% mutation) are carried over from an industrial internship project.

## Repository structure

- `code/uq_industrial.py` — prototype: simulation, model ladder (Linear / Ridge / RandomForest / GPR), calibration analysis, GA λ-sweep
- `code/uq_full_study.py` — full 12-week study: multi-seed robustness, 2-D sensitivity maps, budget table, sample-size / noise ablations, active-learning pre-study
- `results/` — all experiment outputs (`uq_results.json`, `uq_full_results.json`)
- `figures/` — 14 figures (prediction intervals, calibration curve, λ trade-off, Pareto front, seed robustness, active learning, …)
- `docs/UQ_Full_Study_Report.docx` — full research report (Chinese)

## Reproduce

Python ≥ 3.10 with `numpy`, `scikit-learn`, `matplotlib`:

```
python code/uq_industrial.py
python code/uq_full_study.py
```

Adjust the output directory `OUT` at the top of each script before running.

## Data honesty note

Enterprise production data is confidential. All results come from a simulation prototype that replicates the real system's process ranges (melt temperature 1600–2000 °C, drawing speed 5–15 m/s, curing time 10–60 min) and its scoring structure (four-indicator weights 0.4/0.3/0.2/0.1 over thermal conductivity, tensile strength, fiber length, and density).

© 2026 Mingshan Li. Shared for academic review.
