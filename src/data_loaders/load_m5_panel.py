from __future__ import annotations

import os
from typing import Dict

import numpy as np
import pandas as pd


def _resolve_sales_path(base_path: str) -> str:
    sales_candidates = [
        os.path.join(base_path, "sales_train_validation.csv"),
        os.path.join(base_path, "sales_train_evaluation.csv"),
    ]
    sales_path = next((p for p in sales_candidates if os.path.exists(p)), None)
    if sales_path is None:
        raise FileNotFoundError("Missing M5 sales file: sales_train_validation.csv or sales_train_evaluation.csv")
    return sales_path


def load_m5_panel_subset(
    base_path: str = "data/raw/m5",
    num_products: int = 32,
    seed: int = 42,
    state_id: str | None = None,
    store_id: str | None = None,
    cat_id: str | None = None,
    dept_id: str | None = None,
    min_nonzero_days: int = 28,
    max_days: int | None = None,
) -> Dict[str, object]:
    """
    Load a small M5 product panel for graph experiments.

    Returns a dict with:
      - sales: np.ndarray shape (n_products, T)
      - prices: np.ndarray shape (n_products, T)
      - dates: pd.DatetimeIndex length T
      - metadata: pd.DataFrame with one row per product
    """

    cal_path = os.path.join(base_path, "calendar.csv")
    price_path = os.path.join(base_path, "sell_prices.csv")
    sales_path = _resolve_sales_path(base_path)

    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"Missing file: {cal_path}")
    if not os.path.exists(price_path):
        raise FileNotFoundError(f"Missing file: {price_path}")

    calendar = pd.read_csv(cal_path, usecols=["d", "date", "wm_yr_wk"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    prices = pd.read_csv(price_path)
    sales_df = pd.read_csv(sales_path)

    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No d_ columns found in M5 sales file.")

    work = sales_df.copy()
    if state_id is not None:
        work = work[work["state_id"] == state_id]
    if store_id is not None:
        work = work[work["store_id"] == store_id]
    if cat_id is not None:
        work = work[work["cat_id"] == cat_id]
    if dept_id is not None:
        work = work[work["dept_id"] == dept_id]
    if work.empty:
        raise ValueError("No M5 rows matched the requested filters.")

    signal = (work[d_cols].astype(float).to_numpy() > 0).sum(axis=1)
    work = work.assign(nonzero_days=signal)
    work = work[work["nonzero_days"] >= int(min_nonzero_days)].copy()
    if work.empty:
        raise ValueError("No M5 rows remained after min_nonzero_days filter.")

    rng = np.random.default_rng(seed)
    if len(work) > int(num_products):
        chosen_idx = rng.choice(work.index.to_numpy(), size=int(num_products), replace=False)
        work = work.loc[chosen_idx].copy()
    work = work.sort_values(["cat_id", "dept_id", "store_id", "item_id"]).reset_index(drop=True)

    if max_days is not None and int(max_days) > 0:
        d_cols = d_cols[-int(max_days) :]
        calendar = calendar[calendar["d"].isin(d_cols)].copy()

    calendar = calendar.sort_values("date").reset_index(drop=True)
    d_cols = [d for d in d_cols if d in set(calendar["d"])]
    if not d_cols:
        raise ValueError("No usable calendar days found after filtering.")

    selected_price_keys = set(zip(work["store_id"], work["item_id"]))
    prices = prices[
        prices.apply(lambda row: (row["store_id"], row["item_id"]) in selected_price_keys, axis=1)
    ].copy()

    dates = pd.DatetimeIndex(calendar["date"])
    sales_panel = []
    price_panel = []
    meta_rows = []

    for _, row in work.iterrows():
        series = row[d_cols].astype(float).to_numpy(dtype=float)
        ts = pd.DataFrame({"d": d_cols, "sales": series})
        ts = ts.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")

        price_sub = prices[
            (prices["item_id"] == row["item_id"]) & (prices["store_id"] == row["store_id"])
        ][["wm_yr_wk", "sell_price"]].drop_duplicates("wm_yr_wk")
        ts = ts.merge(price_sub, on="wm_yr_wk", how="left")
        ts["sell_price"] = ts["sell_price"].ffill().bfill().fillna(float(np.nanmean(ts["sell_price"])))
        ts["sell_price"] = ts["sell_price"].fillna(0.0)

        sales_panel.append(ts["sales"].to_numpy(dtype=float))
        price_panel.append(ts["sell_price"].to_numpy(dtype=float))
        meta_rows.append(
            {
                "id": row["id"],
                "item_id": row["item_id"],
                "dept_id": row["dept_id"],
                "cat_id": row["cat_id"],
                "store_id": row["store_id"],
                "state_id": row["state_id"],
                "nonzero_days": int(row["nonzero_days"]),
                "mean_sales": float(np.mean(series)),
                "zero_rate": float(np.mean(series == 0)),
            }
        )

    metadata = pd.DataFrame(meta_rows)
    return {
        "sales": np.asarray(sales_panel, dtype=np.float32),
        "prices": np.asarray(price_panel, dtype=np.float32),
        "dates": dates,
        "metadata": metadata,
    }
