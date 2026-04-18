# Deep Models as Challengers, Not the Core Method

## Purpose

This note defines how deep learning forecasters should be incorporated into the paper. The goal is to show that modern deep architectures were considered and evaluated, while also explaining why they are not the most appropriate primary solution for highly zero-inflated retail demand.

## Recommended Position in the Paper

Deep models should be included as strong challenger methods rather than as the central forecasting strategy. This is important for two reasons:

1. These models are widely known in the forecasting literature and reviewers may expect them to appear in the benchmark set.
2. Our research question is not whether deep learning can be applied in general, but whether a regime-aware switching strategy is more suitable for structurally heterogeneous retail series, especially when intermittent demand is present.

## Models to Include

### Primary deep challengers

These are the most practical and defensible deep baselines for the current project:

- `LSTM`
- `GRU`
- `TCN`
- `DeepAR`

### Optional extended challengers

These can be added if time and implementation stability allow, but they should not delay the core experiments:

- `Informer`
- `Temporal Fusion Transformer (TFT)`
- `N-BEATS`
- `LSTM-Transformer`

## Why Deep Models Are Not the Main Method Here

The main limitation is structural rather than purely computational. Many of the target retail series in this project are sparse, intermittent, and zero-inflated. Under those conditions:

- the occurrence of demand is often more important than smooth quantity extrapolation
- long zero runs reduce the effective signal available for sequence learning
- model capacity can be wasted fitting noise instead of stable temporal structure
- multi-parameter deep networks typically require richer covariates, more tuning, and more homogeneous patterns than sparse item-level retail series provide

As a result, deep models can be competitive in dense and signal-rich regimes, but they are often less reliable in intermittent or lumpy regimes where specialized intermittent-demand methods or regime-aware switching are better aligned with the data generating process.

## Core Argument for the Paper

The paper should not claim that deep models are weak in general. Instead, it should make the narrower and more defensible claim:

> Deep architectures are powerful benchmark models for nonlinear forecasting, but in highly zero-inflated retail demand they are not always the most suitable primary solution. Their performance tends to depend strongly on signal richness and regime stability. This motivates the use of a regime-aware switching framework that routes each series, or each period, toward a more appropriate forecasting model.

## Experimental Design

### Model groups

Use three benchmark groups:

1. Classical dense-demand baselines
   - `SARIMAX`

2. Intermittent-demand specialists
   - `HURDLE`
   - `TSB`
   - `Croston-SBA`

3. Deep challengers
   - `LSTM`
   - `GRU`
   - `TCN`
   - `DeepAR`

### Evaluation logic

Run comparisons at two levels:

1. Global comparison across all selected products
2. Regime-wise comparison across:
   - stable / dense
   - variable
   - intermittent
   - lumpy

This structure is critical. A global average alone can hide the main result, which is that different models dominate in different demand regimes.

### Metrics

Use at least:

- `MAE`
- `RMSE`
- `sMAPE`

If possible, also add a demand-occurrence oriented metric for sparse series, because zero-versus-nonzero prediction quality is central in intermittent retail demand.

## Expected Interpretation

The expected outcome is not that one model wins everywhere. Instead:

- `SARIMAX` or some deep models may perform well in stable and signal-rich regimes
- `TSB`, `Croston-SBA`, or `HURDLE` should perform better in intermittent and zero-heavy regimes
- the switching regime framework should outperform single-model alternatives overall because it adapts the forecasting strategy to the underlying demand regime

This interpretation supports the main methodological contribution of the paper.

## Practical Recommendation

For the first paper-ready comparison, prioritize the following shortlist:

- `LSTM`
- `GRU`
- `TCN`
- `DeepAR`

This set is broad enough to satisfy the expectation of comparing against modern deep learning methods, while still keeping the experimental workload manageable.

`Informer`, `TFT`, `N-BEATS`, and `LSTM-Transformer` can be discussed as additional modern architectures, but they should be treated as secondary extensions unless there is enough time for careful tuning and stable reproduction.

## Paper-Ready Paragraph

Deep learning architectures such as LSTM, GRU, TCN, and DeepAR were included as challenger models because they are widely recognized for nonlinear sequence forecasting and multihorizon prediction. However, the demand series studied here exhibit strong zero inflation, intermittency, and regime heterogeneity, which reduce the effective learning signal available to high-capacity neural forecasters. For this reason, deep models were used as important comparison baselines, but not as the core methodological choice. Instead, the proposed regime-aware switching framework is designed to exploit the fact that dense, variable, and intermittent demand regimes are better served by different forecasting families.

## Short Spanish Summary for Discussion

- Si conviene incluir deep models porque son benchmarks conocidos.
- No conviene ponerlos como metodo central porque estas series tienen muchos ceros e intermitencia.
- La comparativa debe mostrar que no existe un solo ganador universal.
- El valor del paper esta en demostrar que el mejor modelo depende del regimen.
- Por eso los deep models funcionan como retadores, mientras que el switching regime framework es la propuesta principal.
