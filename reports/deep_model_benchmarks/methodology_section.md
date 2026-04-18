# Methodology Section Draft

## Deep Learning Benchmark Design

To position the proposed regime-aware forecasting framework against widely recognized neural forecasting methods, we included a set of deep learning challenger models in the experimental benchmark. These models were not treated as the main methodological contribution, but rather as strong comparison baselines representing modern nonlinear sequence forecasting approaches.

The benchmark was conducted across three retail datasets with different demand characteristics: `M5 / Walmart`, `Favorita`, and `Amazon`. This multi-dataset design was important to avoid drawing conclusions from a single market environment and to test whether model suitability changes across retail contexts with different levels of sparsity, variability, and intermittency.

The deep learning benchmark focused on four primary challenger models: `LSTM`, `GRU`, `TCN`, and `DeepAR`. These architectures were selected because they are among the most established neural forecasting approaches for temporal data and are sufficiently representative of recurrent, convolutional, and probabilistic deep forecasting families. Additional architectures such as `Informer`, `Temporal Fusion Transformer`, `N-BEATS`, and `LSTM-Transformer` were considered secondary extensions and were not prioritized in the main paper experiments.

## Comparative Model Families

The full benchmark included three model families:

1. `SARIMAX` as a classical dense-demand baseline.
2. `HURDLE`, `TSB`, and `Croston-SBA` as intermittent-demand specialists.
3. `LSTM`, `GRU`, `TCN`, and `DeepAR` as deep learning challenger models.

This structure allowed us to compare classical forecasting, intermittent-demand modeling, and deep learning under the same experimental setting.

## Regime-Aware Evaluation

Rather than relying only on aggregate error across all items, the comparison was organized around demand regimes. Products from each dataset were selected to represent stable, variable, intermittent, and lumpy demand behaviors. This was necessary because retail demand in the three datasets is structurally heterogeneous, and a single forecasting model is unlikely to be optimal for all series.

For each selected product, forecasting performance was evaluated both globally and within regime categories. This design made it possible to test whether deep learning models remain competitive when demand is dense and signal-rich, and whether their performance weakens when the series becomes sparse, zero-inflated, or highly intermittent.

## Why Deep Models Were Used as Challengers

Deep architectures are powerful tools for learning nonlinear temporal dependencies and multihorizon forecasting patterns. However, their effectiveness typically depends on the presence of sufficiently rich and stable signal. In retail demand settings with long zero runs, low effective sample size, and strong regime shifts, high-capacity neural models may struggle to learn robust item-level structure. In contrast, intermittent-demand models and regime-aware routing strategies can better match the structural properties of such series.

For this reason, deep models were included to ensure a rigorous and contemporary comparison, but they were not used as the core forecasting philosophy of the study. The central hypothesis of the paper is that the most appropriate model depends on the demand regime, and that a switching regime framework can exploit this dependence more effectively than any single-model alternative.

## Evaluation Metrics

Forecast accuracy was measured using standard error metrics including `MAE`, `RMSE`, and `sMAPE`. Because sparse retail demand is strongly influenced by whether demand occurs at all, the analysis also recommends including a demand-occurrence oriented metric when possible. This is particularly relevant for the `Amazon` dataset and for intermittent product subsets in `M5` and `Favorita`.
