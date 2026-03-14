# Exploratory Analysis of Demand Regimes

## Experimental Setup
We followed a five-step exploratory protocol: (1) choose category, (2) sample 10-15 products, (3) build time-series dataset, (4) visualize demand trajectories, and (5) identify demand regimes.

Category alignment across datasets:
- M5 Walmart: `HOUSEHOLD`
- Favorita: `CLEANING` (store 1)
- Amazon 2023: `Health_and_Household`

We sampled 15 product-level series per dataset and computed regime descriptors per series:
- `zero_rate` (share of zero-demand days)
- `ADI` (average demand interval)
- `CV^2` (demand variability over non-zero days)
- `transition_score` (distributional shift between first and second half)

Regime labels:
- `stable`: lower intermittency and low shift
- `intermittent`: high intermittency, low shift
- `transition`: clear shift in demand pattern

## Main Results
Counts by regime:
- AMAZON_2023: 7 intermittent, 8 transition
- FAVORITA: 14 stable, 1 transition
- M5_WALMART: 6 intermittent, 9 transition

Dataset-level means:
- AMAZON_2023: zero_rate=0.923, ADI=18.166, CV^2=0.206, transition_score=1.583
- FAVORITA: zero_rate=0.075, ADI=1.084, CV^2=0.478, transition_score=0.272
- M5_WALMART: zero_rate=0.765, ADI=6.822, CV^2=0.250, transition_score=2.020

Interpretation:
- Amazon and M5 exhibit sparse/intermittent behavior with frequent regime shifts.
- Favorita cleaning series are mostly stable under the current aggregation and store filter.
- Regime behavior differs substantially by dataset, supporting regime-aware forecasting design.

## Suggested Figures (from generated plots)
Use representative examples from:
- `reports/regime_eda_final/plots/`

Recommended picks:
- M5 transition: `M5_WALMART__HOUSEHOLD_1_187_WI_1_validation.png`
- M5 intermittent: `M5_WALMART__HOUSEHOLD_1_015_CA_3_validation.png`
- Amazon transition: `AMAZON_2023__B072KDRZH5.png`
- Amazon intermittent: `AMAZON_2023__B00QWO9P0O.png`
- Favorita stable: `FAVORITA__item_268443.png`
- Favorita transition: `FAVORITA__item_789224.png`

## Reproducibility
Command used:

```powershell
.\.venv\Scripts\python.exe -m src.experiments.run_regime_eda --datasets m5,favorita,amazon --n-products 15 --amazon-file Health_and_Household.jsonl.gz --amazon-max-rows 100000 --out-dir reports/regime_eda_final
```
