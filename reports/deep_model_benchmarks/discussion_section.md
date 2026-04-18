# Discussion Section Draft

## Interpreting Deep Model Results

The inclusion of `LSTM`, `GRU`, `TCN`, and `DeepAR` in the benchmark was important because these are among the most recognized deep forecasting models in current practice. Their presence strengthens the study by showing that the proposed framework was not evaluated only against classical statistical baselines, but also against modern neural approaches.

However, the experimental results should be interpreted in relation to the structure of the demand series rather than in terms of a universal ranking. Across the three datasets, the main pattern to investigate is whether deep learning models perform well when the demand signal is dense and comparatively stable, but lose effectiveness when the series becomes sparse, intermittent, or strongly zero-inflated.

## Why Deep Models May Underperform in This Setting

A likely explanation is that many item-level retail series do not provide the kind of rich, continuous signal that high-capacity deep models typically exploit most effectively. In the presence of long zero runs, abrupt regime changes, and limited nonzero observations, neural sequence models may devote a large portion of their capacity to fitting irregular noise rather than learning persistent structure.

This issue is especially important in the `Amazon` dataset, where intermittent and low-rotation behavior is more common, but it may also appear in selected products from `M5` and `Favorita`. Therefore, weaker performance from deep models in this study should not be interpreted as evidence against deep forecasting in general, but rather as evidence that model suitability is highly dependent on the demand regime and the effective information content of the series.

## Why the Switching Regime Framework Is Better Aligned

The main contribution of the proposed approach is that it does not assume that one model family should dominate everywhere. Instead, it explicitly recognizes that different demand regimes favor different forecasting mechanisms. Dense and stable series may be well served by models such as `SARIMAX` or some deep architectures, whereas intermittent and lumpy series may be better handled by `TSB`, `Croston-SBA`, or `HURDLE`.

If this pattern appears consistently across `M5`, `Favorita`, and `Amazon`, then the empirical support for the switching regime framework becomes substantially stronger. In that case, the contribution of the paper is not to replace deep learning with a single alternative model, but to demonstrate that a regime-aware strategy can integrate the strengths of different forecasting families under heterogeneous retail demand.

## Suggested Closing Paragraph

Overall, the benchmark with deep learning challenger models reinforces the central claim of the paper: modern neural forecasters are valuable comparison methods, but they are not universally optimal in zero-inflated retail demand. Their strengths are concentrated in signal-rich settings, whereas intermittent-demand specialists and regime-aware routing become more appropriate as sparsity and structural heterogeneity increase. This supports the use of a switching regime forecasting framework as a more flexible and data-aligned solution across diverse retail datasets.
