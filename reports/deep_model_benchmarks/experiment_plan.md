# Deep Model Benchmark Experiment Plan

## Objective

Evaluate modern deep forecasting models as challenger methods against the regime-aware framework using all three project datasets:

- `M5 / Walmart`
- `Favorita`
- `Amazon`

The goal is not to prove that deep learning is universally inferior, but to test whether its performance changes across demand regimes and whether a switching regime strategy is more appropriate overall.

## Datasets

### 1. M5 / Walmart

Use product series from household-related categories already aligned with the current experiments. This dataset is useful because it includes dense, variable, and intermittent behavior depending on the item.

### 2. Favorita

Use grocery and cleaning product families with different demand densities. Favorita is important because it provides a retail setting with stronger calendar and promotional structure.

### 3. Amazon

Use `Health_and_Household` product series. This dataset is especially relevant for sparse and intermittent retail demand and is therefore critical for testing the limitations of deep models under zero inflation.

## Benchmark Groups

### Classical baseline

- `SARIMAX`

### Intermittent-demand specialists

- `HURDLE`
- `TSB`
- `Croston-SBA`

### Deep challengers

- `LSTM`
- `GRU`
- `TCN`
- `DeepAR`

### Optional extended deep models

- `Informer`
- `TFT`
- `N-BEATS`
- `LSTM-Transformer`

## Product Selection

For each dataset, select products that cover at least four structural behaviors:

- stable / dense demand
- variable demand
- intermittent demand
- lumpy demand

Recommended initial scale:

- `10 to 15 products per dataset`
- balanced across regimes as much as possible

This gives a total benchmark pool large enough to support cross-dataset comparison while remaining manageable for paper experiments.

## Evaluation Structure

### Level 1. Global benchmark

Compare all models on the full selected pool across the three datasets.

### Level 2. Regime-wise benchmark

Compare the same models inside each regime:

- stable
- variable
- intermittent
- lumpy

### Level 3. Cross-dataset interpretation

Study whether model behavior is consistent or dataset-dependent:

- Does `TCN` help more in `Favorita` than in `Amazon`?
- Do `LSTM/GRU` improve in denser `M5` products but fail in sparse `Amazon` items?
- Do intermittent specialists remain stronger in the zero-heavy cases across all three datasets?

## Metrics

Use at least:

- `MAE`
- `RMSE`
- `sMAPE`

If possible, add one demand-occurrence metric for sparse cases, because zero-versus-nonzero prediction quality is central in intermittent retail demand.

## Main Hypothesis

The main expected result is:

1. Deep models can be competitive in signal-rich and stable series.
2. Their performance degrades in sparse and highly zero-inflated regimes.
3. The best model depends on the demand regime and sometimes also on the dataset.
4. A regime-aware switching framework should therefore outperform single-model strategies on average across the full heterogeneous benchmark.

## Recommended Execution Order

1. Run the current benchmark set on all three datasets:
   - `SARIMAX`
   - `HURDLE`
   - `TSB`
   - `Croston-SBA`

2. Add the primary deep challengers:
   - `LSTM`
   - `GRU`
   - `TCN`
   - `DeepAR`

3. Aggregate results:
   - by dataset
   - by regime
   - overall

4. Compare against the regime-switching method.

## Expected Paper Message

Using three datasets is important because it prevents the conclusion from being tied to a single retail environment. If deep models underperform mainly in sparse and intermittent regimes across `M5`, `Favorita`, and `Amazon`, then the argument for the proposed switching regime approach becomes much stronger and more generalizable.
