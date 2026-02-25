import os
import numpy as np
import pandas as pd


def load_favorita_series(
    base_path="data/raw/favorita",
    store_nbr=1,
    family="GROCERY I",
    chunksize=1_000_000,
) -> pd.DataFrame:
    """
    Builds a daily time series for Favorita using chunked reading (train.csv is huge).
    Output columns: date, sales, price
    """

    train_path = os.path.join(base_path, "train.csv")
    items_path = os.path.join(base_path, "items.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing file: {train_path}")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Missing file: {items_path}")

    items = pd.read_csv(items_path, usecols=["item_nbr", "family"])
    wanted_items = set(items.loc[items["family"] == family, "item_nbr"].astype("int64"))

    if not wanted_items:
        raise ValueError(f"No items found for family='{family}' in items.csv")

    usecols = ["date", "store_nbr", "item_nbr", "unit_sales", "onpromotion"]
    dtypes = {
        "store_nbr": "int16",
        "item_nbr": "int32",
        "unit_sales": "float32",
    }

    daily_parts = []
    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["date"],
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = chunk[chunk["store_nbr"] == store_nbr]
        if chunk.empty:
            continue

        chunk = chunk[chunk["item_nbr"].isin(wanted_items)]
        if chunk.empty:
            continue

        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)

        part = chunk.groupby("date", as_index=False).agg(
            sales=("unit_sales", "sum"),
            promo=("onpromotion", "sum"),
        )
        daily_parts.append(part)

    if not daily_parts:
        return pd.DataFrame(columns=["date", "sales", "price"])

    daily = pd.concat(daily_parts, ignore_index=True).groupby("date", as_index=False).sum()
    daily = daily.sort_values("date")

    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
    daily["sales"] = daily["sales"].fillna(0.0)
    daily["price"] = np.nan

    return daily[["date", "sales", "price"]]
