# Paper Experiments

This section groups notebook workflows that are intended to be more
paper-ready than the exploratory notebooks in the root `notebooks/`
folder.

## Goal

Organize a clear regime-aware forecasting story around:

- high-demand / stable items -> `SARIMAX`
- transition / mixed-demand items -> `Hurdle`
- intermittent / very-low-demand items -> `TSB`

## Proposed Case Studies

These cases are grounded in the regime EDA artifacts already selected in:

- `reports/regime_eda_final/section_final.md`
- `reports/regime_eda_final/plots/`

Recommended five-item set:

1. `FAVORITA__item_268443`
   - expected role: high-demand stable case
   - preferred model: `SARIMAX`
2. `FAVORITA__item_789224`
   - expected role: transition case
   - preferred model: `Hurdle`
3. `M5_WALMART__HOUSEHOLD_1_187_WI_1_validation`
   - expected role: transition retail case
   - preferred model: `Hurdle`
4. `AMAZON_2023__B00QWO9P0O`
   - expected role: low-demand intermittent case
   - preferred model: `TSB`
5. `AMAZON_2023__B0BZTLKX7H`
   - expected role: severe intermittent stress test
   - preferred model: `TSB`

## Notebooks

- `21_regime_paper_cases.ipynb`
  - defines the five paper cases and links each one to the model it should
    highlight
- `22_regime_paper_storyboard.ipynb`
  - turns the five cases into a paper narrative: regime diagnosis,
    model choice, and figure/table plan

## Notes

- The current production selector in `src/models/regime_forecast_engine.py`
  chooses between `SARIMAX` and `Hurdle`.
- `TSB` is available in `src/models/tsb_model.py` and should be presented as
  a focused intermittent-demand experiment.
