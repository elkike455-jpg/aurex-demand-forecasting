# Exploratory Analysis of Demand Regimes

## Setup
Category alignment used in this experiment:
- M5 Walmart: `HOUSEHOLD`
- Favorita: `CLEANING` (store 1)
- Amazon 2023: `Health_and_Household`

For each dataset, 15 product-level series were selected and analyzed with:
- `zero_rate`
- `ADI`
- `CV^2`
- `transition_score`

Regime labels:
- `stable`
- `intermittent`
- `transition`

## Results
Regime counts:
- AMAZON_2023: 7 intermittent, 8 transition
- FAVORITA: 14 stable, 1 transition
- M5_WALMART: 6 intermittent, 9 transition

Mean indicators by dataset:
- AMAZON_2023: zero_rate=0.923, ADI=18.166, CV^2=0.206, transition_score=1.583
- FAVORITA: zero_rate=0.075, ADI=1.084, CV^2=0.478, transition_score=0.272
- M5_WALMART: zero_rate=0.765, ADI=6.822, CV^2=0.250, transition_score=2.020

Interpretation:
- Amazon and M5 show sparse/intermittent behavior with frequent regime shifts.
- Favorita cleaning series are mostly stable under the selected store/category granularity.
- The presence of multiple regimes supports regime-aware forecasting.

## Figure Set (6 Recommended)
Selected and copied to:
`reports/paper_figures/regime_eda_final/`

- `M5_intermittent.png`: M5 intermittent example (`HOUSEHOLD_1_015_CA_3_validation`)
- `M5_transition.png`: M5 transition example (`HOUSEHOLD_1_187_WI_1_validation`)
- `Amazon_intermittent.png`: Amazon intermittent example (`B00QWO9P0O`)
- `Amazon_transition.png`: Amazon transition example (`B072KDRZH5`)
- `Favorita_stable.png`: Favorita stable example (`item_268443`)
- `Favorita_transition.png`: Favorita transition example (`item_789224`)

## Reproducibility
Command:
```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_regime_eda --datasets m5,favorita,amazon --n-products 15 --amazon-file Health_and_Household.jsonl.gz --amazon-max-rows 100000 --out-dir reports/regime_eda_final
```

Generated artifacts:
- `reports/regime_eda_final/regime_summary.csv`
- `reports/regime_eda_final/regime_counts.csv`
- `reports/regime_eda_final/plots/`
