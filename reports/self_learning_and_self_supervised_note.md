# Self-Learning and Self-Supervised Learning

## What self-learning means

Self-learning is a broad idea: a system improves by using its own experience, predictions, feedback, or newly observed data after initial training. In forecasting, this can mean periodic retraining, online updates, pseudo-labeling, active error monitoring, or letting the model adapt its graph or parameters as new demand history arrives.

Practical retail example: an AUREX forecasting pipeline could retrain weekly as new M5-style sales arrive, update product similarity edges when demand correlations shift, or flag products whose forecast errors suggest a new demand regime.

## What self-supervised learning means

Self-supervised learning is a training strategy where labels are created from the data itself instead of manually annotated. The model solves a pretext task that forces it to learn useful representations.

Practical time-series and graph examples:

- mask part of a sales history and train the model to reconstruct it;
- predict the next block of demand from prior context;
- contrast two augmented views of the same product time series;
- learn product embeddings by predicting whether two products share category, store, or correlated demand behavior;
- mask graph edges and train the model to recover likely product relationships.

## Difference

Self-learning is about how a system keeps improving or adapting over time.

Self-supervised learning is about how a model gets a training signal without external labels.

They can be combined: a graph forecasting model can use self-supervised pretraining to learn product embeddings, then self-learn by updating those embeddings or graph edges as new sales data arrives.

## Why this matters for graph demand forecasting

Retail demand graphs are rarely fixed. Product relationships change with seasonality, assortment changes, price movement, holidays, SNAP effects, and demand shocks. A static graph built once from historical correlation may become stale.

Self-supervised learning could help by learning better node representations before supervised forecasting. This is useful when direct labels such as promotions, brand, display placement, or product substitutability are missing.

Self-learning could help by refreshing the graph and model as new demand arrives, especially for intermittent and low-volume products where one unusual demand event can change the useful neighborhood.

## Relevance to AUREX / ASG-RNN-like work

For the current M5 setup, a practical next step is:

1. Pretrain product embeddings from M5 metadata, prices, sales windows, and graph-neighbor prediction.
2. Use those embeddings as static node features in the ASG-RNN-like graph-recurrent model.
3. Periodically rebuild the hybrid graph using recent demand and price behavior.
4. Track whether graph refreshes improve MAE, variance ratio, trend correlation, and peak detection on the benchmark products.

This would keep the current implementation honest: M5 does not provide explicit promotions or rich product attributes, but self-supervised graph/time-series tasks can extract more structure from the fields M5 does provide.
