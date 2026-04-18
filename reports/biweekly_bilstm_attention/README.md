# Biweekly BiLSTM Attention Forecast

Pipeline: 28-day lookback, 14-day horizon, 26 rolling validation windows.
Models: BiLSTM-HighDemand and BiLSTM-Intermittent.
Architecture: BiLSTM(128, return sequences) + dropout + temporal attention + BiLSTM(64) + Dense(32 ReLU) + Dense(14).
Peak definition: predicted or actual sales > rolling_mean_28d * 1.5.
Intermittent model: weighted Huber loss with 3x under-predicted peak penalty and early stopping on validation PDR.

Outputs:
- biweekly_metrics_all.csv
- biweekly_predictions_all.csv
- aggregate_metrics_summary.csv
- figures/*
- models/*_bilstm_attention.pt