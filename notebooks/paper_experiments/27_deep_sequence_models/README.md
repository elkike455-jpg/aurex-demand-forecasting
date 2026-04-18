# Deep Sequence Models Track

This folder is intentionally separated from [26_five_case_regime_experiments.ipynb](/c:/Users/braya/Documents/Research/aurex-demand-forecasting-main/notebooks/paper_experiments/26_five_case_regime_experiments.ipynb).

Design decision:

- `26_five_case_regime_experiments.ipynb` remains the main paper notebook
- this `27_deep_sequence_models/` folder is a parallel experiment track
- deep models should be developed here without editing or restructuring notebook 26

Recommended structure:

- `27_deep_sequence_model_experiments.ipynb`
  Paper-facing master notebook for the deep-model comparison
- `scratch/`
  Optional support notebooks for model-specific debugging or implementation trials

Why one master notebook instead of many:

- it mirrors the narrative style of notebook 26
- it keeps the cross-model comparison in one place
- it avoids scattering paper results across several notebooks

When to add extra notebooks:

- if `TCN`, `Informer`, `TFT`, `N-BEATS`, or `LSTM-Transformer` need architecture-specific debugging
- if a model requires long exploratory work that would make the master notebook noisy

Current scope:

- Implemented now: `LSTM`, `GRU`, `DeepAR`, `TCN`, `Informer`, `TFT`, `N-BEATS`, `LSTM-Transformer`
- Baselines kept for evidence: `SARIMAX`, `HURDLE`, `TSB`, `CROSTON_SBA`

Core goals:

- use the same representative case logic as notebook 26
- include the three datasets: `M5_WALMART`, `FAVORITA`, `AMAZON_2023`
- preserve a clean, reproducible notebook for paper-ready plots and tables
