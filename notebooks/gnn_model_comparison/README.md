# GNN Model Comparison

This folder is dedicated to notebook work for:

- comparing forecasting models under a shared evaluation setup
- testing graph-based forecasting ideas and Graph Neural Networks
- keeping GNN experiments separate from the general notebook backlog

## Suggested Notebook Flow

1. `01_benchmark_scope.ipynb`
   Define datasets, target products/series, forecast horizon, and metrics.
2. `02_baseline_model_comparison.ipynb`
   Compare classical, intermittent, and deep baseline models.
3. `03_graph_construction.ipynb`
   Build product or series graphs from similarity, co-movement, or metadata.
4. `04_gnn_benchmark.ipynb`
   Train and evaluate graph neural forecasting candidates.
5. `05_results_review.ipynb`
   Consolidate tables, plots, and conclusions.

## Project Links

- Existing exploratory GNN notebook:
  - `../29_graph_neural_network.ipynb`
- Existing benchmark runner:
  - `../../src/experiments/run_gnn_benchmark.py`
- Existing graph forecasting model:
  - `../../src/models/graph_forecast_model.py`

## Folder Layout

- `scratch/`
  Temporary notebook drafts and quick tests.
- `figures/`
  Saved plots for model comparison and graph diagnostics.
- `exports/`
  Tables, CSV summaries, and notebook outputs to reuse in reports.

## Notes

- Keep final comparison notebooks here instead of the root `notebooks/` folder.
- Reuse loaders from `src/data_loaders/` so results stay comparable with the rest of the project.
- Prefer storing generated artifacts in `figures/` and `exports/` instead of mixing them beside notebooks.
