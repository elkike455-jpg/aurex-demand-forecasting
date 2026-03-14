# aurex-demand-forecasting

## Exploratory Demand Regimes (Dr. Yow Step)

Final EDA outputs are available in:
- `reports/regime_eda_final/regime_summary.csv`
- `reports/regime_eda_final/regime_counts.csv`
- `reports/regime_eda_final/plots/`
- `reports/regime_eda_final/section_final.md`

Paper-ready figure pack (6 selected examples):
- `reports/paper_figures/regime_eda_final/M5_intermittent.png`
- `reports/paper_figures/regime_eda_final/M5_transition.png`
- `reports/paper_figures/regime_eda_final/Amazon_intermittent.png`
- `reports/paper_figures/regime_eda_final/Amazon_transition.png`
- `reports/paper_figures/regime_eda_final/Favorita_stable.png`
- `reports/paper_figures/regime_eda_final/Favorita_transition.png`

Run command:
```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_regime_eda --datasets m5,favorita,amazon --n-products 15 --amazon-file Health_and_Household.jsonl.gz --amazon-max-rows 100000 --out-dir reports/regime_eda_final
```
