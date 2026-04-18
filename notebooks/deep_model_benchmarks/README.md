# Deep Model Benchmarks

This folder contains notebooks for benchmarking deep challenger models against the regime-aware forecasting approach across the three project datasets:

- `M5 / Walmart`
- `Favorita`
- `Amazon`

## Notebook Order

- `30_deep_benchmark_plan.ipynb`
  Planning notebook for benchmark scope, datasets, regimes, and challenger models.

- `31_triplet_dataset_setup.ipynb`
  Loads one representative series from each dataset and standardizes them into a common structure.

- `32_regime_case_screening.ipynb`
  Computes demand descriptors such as `zero_rate`, `ADI`, and `CV2` to identify stable, variable, intermittent, and lumpy candidate cases.

- `33_deep_challenger_benchmark.ipynb`
  Main benchmark notebook scaffold for comparing classical models, intermittent-demand models, and deep challenger models.

## Suggested Workflow

1. Run `30_deep_benchmark_plan.ipynb`
2. Run `31_triplet_dataset_setup.ipynb`
3. Run `32_regime_case_screening.ipynb`
4. Extend and run `33_deep_challenger_benchmark.ipynb`

## Notes

- These notebooks are intentionally lightweight and project-aligned.
- They reuse loaders from `src/data_loaders/`.
- The benchmark target is to compare `LSTM`, `GRU`, `TCN`, and `DeepAR` as challenger models, not as the core method.
