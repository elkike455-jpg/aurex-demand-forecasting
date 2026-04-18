# Walmart Repo LSTM #2 Assessment

Repository inspected:

- `Walmart-LSTM-Sales-Forecasting-main/Walmart-LSTM-Sales-Forecasting-main`

## Inspection Summary

The external repo is not a clean reusable forecasting package. It is mainly:

- two exploratory notebooks
- one preprocessed Walmart weekly dataset
- one saved Keras BiLSTM model
- one scaler and a few exported prediction artifacts

The original workflow is strongly tied to:

- Walmart weekly store sales
- a precomputed tabular dataset `optimized_dataset.csv`
- extra engineered features and external covariates
- random `train_test_split(...)` instead of chronological time split
- local author-specific Windows paths hardcoded inside notebooks

## What Was Reused

The reusable part is the model architecture itself:

- Keras `Sequential`
- `Bidirectional(LSTM(50, return_sequences=True))`
- `Dropout(0.2)`
- `Bidirectional(LSTM(50, return_sequences=False))`
- `Dropout(0.2)`
- `Dense(1)`
- `optimizer='adam'`
- `loss='mean_squared_error'`

## What Had To Be Adapted

To make it fit the unified M5 benchmark, the following changes were necessary:

- switch from tabular Walmart rows to single-series M5 demand sequences
- replace random train/test split with chronological `309 / 28 / 28`
- replace the repo's mixed-feature tabular input with recursive univariate windows
- fit scaling using the benchmark protocol instead of the repo's full-dataset tabular scaling
- generate recursive multi-step forecasts for the final 28-day test horizon

## Usability Verdict

Verdict: **partially usable, but not directly reusable as-is**.

Why:

- the original repo is too Walmart-specific and notebook-centric
- its evaluation design is not suitable for a fair time-series benchmark
- its data pipeline is not transferable to the single-series M5 benchmark without rewriting the surrounding workflow

However:

- the BiLSTM architecture itself can be treated as a valid external implementation after adaptation
- in this project it should be considered **LSTM implementation #2 only as an adapted external architecture**, not as a plug-and-play benchmark repo
