# Common Train/Test Protocol

Official protocol for all future model comparisons on:

- the same 16 M5 products used in `notebooks/29_graph_neural_network.ipynb`
- the three official benchmark products selected from that subset

## Official Configuration

- dataset scope: `M5_GNN_SUBSET`
- product subset: same `16` products from `notebook29`
- benchmark product scope:
  - `FOODS_3_228_CA_1_validation`
  - `FOODS_2_044_CA_3_validation`
  - `HOBBIES_1_133_CA_4_validation`
- state filter: `CA`
- random seed for subset reproduction: `42`
- analysis window: latest `365` days
- minimum activity filter for subset construction: `28` non-zero days
- input context length: `28` days
- validation horizon: `28` days
- test horizon: `28` days

## Split Logic

Use one fixed chronological split for every model:

1. `train`
   - all observations before the last `56` days
2. `validation`
   - the `28` days immediately before the test window
3. `test`
   - the final `28` days

For a `365`-day series, that means:

- `train`: first `309` days
- `validation`: next `28` days
- `test`: final `28` days

This avoids leakage and keeps every model on the same forecast horizon.

## Official Preprocessing

Apply the same preprocessing whenever the model family allows it:

- use raw daily sales in chronological order
- do not shuffle time steps
- replace `NaN` and non-finite values with `0.0` only when required by the implementation
- fit any scaler or normalization step on the `train` segment only
- use `train + validation` only for the final refit after hyperparameter choices are fixed
- evaluate every model on the same held-out `28`-day `test` block

## Fair-Comparison Rules

- same benchmark products
- same date window
- same split boundaries
- same forecast horizon
- same reporting metrics per product
- no model may inspect future values from validation or test during training

## Experimental Parameters To Document

Every future benchmark run should report at least:

- `dataset_name`
- `series_id`
- `seed`
- `max_days`
- `context_length`
- `val_days`
- `test_days`
- whether the model was refit on `train + validation`
- any scaling rule used
- model-specific hyperparameters such as epochs, hidden size, or graph settings

## Reference Implementation

Reusable helpers were added in:

- `src/experiments/common_protocol.py`

Core helpers:

- `OFFICIAL_BENCHMARK_PROTOCOL`
- `split_series_protocol(...)`
- `split_panel_protocol(...)`
- `sanitize_series(...)`

## Why This Protocol

This protocol keeps later comparisons consistent with the GNN setup already established in `notebook29`, instead of mixing it with older repo benchmarks that use much longer `365`-day test windows. For the M5 GNN subset, the official horizon is now the final `28` days.
