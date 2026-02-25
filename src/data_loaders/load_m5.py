import os
import numpy as np
import pandas as pd

def load_m5_single_series(
    base_path="data/raw/m5",
    random_pick=True,
    seed=42,
    series_id=None
) -> pd.DataFrame:
    """
    Loads ONE M5 series (one item/store id row) and returns a daily DF:
    columns: date, sales, price

    You must have:
      - sales_train_validation.csv (or sales_train_evaluation.csv)
      - calendar.csv
      - sell_prices.csv
    inside data/raw/m5/
    """

    cal_path = os.path.join(base_path, "calendar.csv")
    price_path = os.path.join(base_path, "sell_prices.csv")

    # accept either file name
    sales_candidates = [
        os.path.join(base_path, "sales_train_validation.csv"),
        os.path.join(base_path, "sales_train_evaluation.csv"),
    ]
    sales_path = next((p for p in sales_candidates if os.path.exists(p)), None)
    if sales_path is None:
        raise FileNotFoundError("Missing M5 sales file: sales_train_validation.csv or sales_train_evaluation.csv")

    calendar = pd.read_csv(cal_path)
    prices = pd.read_csv(price_path)
    sales_df = pd.read_csv(sales_path)

    calendar["date"] = pd.to_datetime(calendar["date"])
    d_cols = [c for c in sales_df.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No d_ columns found in M5 sales file.")

    rng = np.random.default_rng(seed)

    if series_id is not None:
        row = sales_df[sales_df["id"] == series_id]
        if row.empty:
            raise ValueError(f"series_id not found: {series_id}")
        row = row.iloc[0]
    else:
        if not random_pick:
            row = sales_df.iloc[0]
        else:
            # pick a row with some signal and with prices available
            for _ in range(2000):
                idx = int(rng.integers(0, len(sales_df)))
                r = sales_df.iloc[idx]
                y = r[d_cols].astype(float).values
                if np.all(y == 0):
                    continue
                item_id, store_id = r["item_id"], r["store_id"]
                if prices[(prices["item_id"] == item_id) & (prices["store_id"] == store_id)].empty:
                    continue
                row = r
                break
            else:
                raise RuntimeError("Could not find a valid random M5 series with prices.")

    item_id, store_id = row["item_id"], row["store_id"]

    ts = pd.DataFrame({"d": d_cols, "sales": row[d_cols].astype(float).values})
    ts = ts.merge(calendar[["d", "date", "wm_yr_wk"]], on="d", how="left")

    price_sub = prices[(prices["item_id"] == item_id) & (prices["store_id"] == store_id)][["wm_yr_wk", "sell_price"]]
    price_sub = price_sub.drop_duplicates("wm_yr_wk")

    ts = ts.merge(price_sub, on="wm_yr_wk", how="left")
    ts["sell_price"] = ts["sell_price"].ffill().bfill()
    ts = ts.rename(columns={"sell_price": "price"})

    return ts[["date", "sales", "price"]].sort_values("date").reset_index(drop=True)