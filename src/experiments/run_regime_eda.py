import argparse
import gzip
import json
import os
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _resolve_amazon_path(base_path, filename):
    """Resolve either compressed or plain Amazon review files."""
    candidates = [os.path.join(base_path, filename)]

    if filename.endswith(".jsonl.gz"):
        candidates.append(os.path.join(base_path, filename[:-3]))
    elif filename.endswith(".jsonl"):
        candidates.append(os.path.join(base_path, filename + ".gz"))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"Missing Amazon file. Tried: {', '.join(candidates)}")


def _safe_name(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text))


def _compute_regime_metrics(y):
    y = np.asarray(y, dtype=float)
    if len(y) == 0:
        return {
            "n_days": 0,
            "zero_rate": np.nan,
            "adi": np.nan,
            "cv2": np.nan,
            "transition_score": np.nan,
            "sbc_class": "unknown",
            "regime_label": "unknown",
        }

    nonzero_mask = y > 0
    n_nonzero = int(nonzero_mask.sum())
    zero_rate = float(1.0 - (n_nonzero / len(y)))
    adi = float(len(y) / max(1, n_nonzero))

    nz = y[nonzero_mask]
    if len(nz) == 0 or float(np.mean(nz)) <= 1e-12:
        cv2 = float("inf")
    else:
        cv2 = float((np.std(nz) / np.mean(nz)) ** 2)

    half = len(y) // 2
    if half >= 1:
        z1 = float((y[:half] == 0).mean())
        z2 = float((y[half:] == 0).mean())
        m1 = float(np.mean(y[:half])) + 1e-8
        m2 = float(np.mean(y[half:])) + 1e-8
        transition_score = float(abs(z2 - z1) + abs(np.log(m2 / m1)))
    else:
        transition_score = 0.0

    if adi < 1.32 and cv2 < 0.49:
        sbc_class = "smooth"
    elif adi < 1.32 and cv2 >= 0.49:
        sbc_class = "erratic"
    elif adi >= 1.32 and cv2 < 0.49:
        sbc_class = "intermittent"
    else:
        sbc_class = "lumpy"

    if transition_score >= 0.60:
        regime_label = "transition"
    elif adi >= 1.32:
        regime_label = "intermittent"
    else:
        regime_label = "stable"

    return {
        "n_days": int(len(y)),
        "zero_rate": zero_rate,
        "adi": adi,
        "cv2": cv2,
        "transition_score": transition_score,
        "sbc_class": sbc_class,
        "regime_label": regime_label,
    }


def _plot_series(date, y, title, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 3.5))
    plt.plot(date, y, linewidth=1.1)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("sales_proxy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _iter_jsonl(path, max_rows=None):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            if max_rows is not None and i > int(max_rows):
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_m5_category_series(base_path, cat_id="HOUSEHOLD", n_products=12, seed=42):
    cal_path = os.path.join(base_path, "calendar.csv")
    sales_candidates = [
        os.path.join(base_path, "sales_train_validation.csv"),
        os.path.join(base_path, "sales_train_evaluation.csv"),
    ]
    sales_path = next((p for p in sales_candidates if os.path.exists(p)), None)
    if sales_path is None:
        raise FileNotFoundError("Missing M5 sales file.")
    if not os.path.exists(cal_path):
        raise FileNotFoundError("Missing M5 calendar.csv.")

    sales = pd.read_csv(sales_path)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    if not d_cols:
        raise ValueError("No d_ columns in M5 file.")

    sub = sales[sales["cat_id"] == cat_id].copy()
    if sub.empty:
        raise ValueError(f"No rows found for cat_id={cat_id}.")

    rng = np.random.default_rng(seed)
    sub["_sum_sales"] = sub[d_cols].sum(axis=1).values
    sub = sub[sub["_sum_sales"] > 0].copy()
    if sub.empty:
        raise ValueError(f"No non-zero rows for cat_id={cat_id}.")

    if len(sub) > n_products:
        idx = rng.choice(sub.index.values, size=n_products, replace=False)
        pick = sub.loc[idx]
    else:
        pick = sub

    calendar = pd.read_csv(cal_path, usecols=["d", "date"])
    d_to_date = pd.Series(pd.to_datetime(calendar["date"]).values, index=calendar["d"]).to_dict()

    series_map = {}
    for _, row in pick.iterrows():
        sid = row["id"]
        y = row[d_cols].astype(float).values
        date = pd.to_datetime([d_to_date[d] for d in d_cols])
        df = pd.DataFrame({"date": date, "sales": y}).sort_values("date").reset_index(drop=True)
        series_map[sid] = df

    return series_map


def load_favorita_family_series(
    base_path,
    family="CLEANING",
    store_nbr=1,
    n_products=12,
    chunksize=1_000_000,
):
    train_path = os.path.join(base_path, "train.csv")
    items_path = os.path.join(base_path, "items.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Missing {items_path}")

    items = pd.read_csv(items_path, usecols=["item_nbr", "family"])
    wanted_items = set(items.loc[items["family"] == family, "item_nbr"].astype("int64"))
    if not wanted_items:
        raise ValueError(f"No items for family={family}")

    usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}

    item_sum = Counter()
    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["date"],
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = chunk[(chunk["store_nbr"] == store_nbr) & (chunk["item_nbr"].isin(wanted_items))]
        if chunk.empty:
            continue
        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)
        g = chunk.groupby("item_nbr")["unit_sales"].sum()
        for k, v in g.items():
            item_sum[int(k)] += float(v)

    if not item_sum:
        return {}

    selected_items = [k for k, _ in item_sum.most_common(n_products)]
    selected_set = set(selected_items)

    item_daily = defaultdict(lambda: defaultdict(float))
    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["date"],
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = chunk[(chunk["store_nbr"] == store_nbr) & (chunk["item_nbr"].isin(selected_set))]
        if chunk.empty:
            continue
        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)
        grouped = chunk.groupby(["item_nbr", "date"], as_index=False)["unit_sales"].sum()
        for _, row in grouped.iterrows():
            item_daily[int(row["item_nbr"])][pd.Timestamp(row["date"]).normalize()] += float(row["unit_sales"])

    series_map = {}
    for item in selected_items:
        daily = item_daily.get(item, {})
        if not daily:
            continue
        df = pd.DataFrame({"date": list(daily.keys()), "sales": list(daily.values())}).sort_values("date")
        full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
        df["sales"] = df["sales"].fillna(0.0).astype(float)
        series_map[f"item_{item}"] = df[["date", "sales"]]

    return series_map


def load_amazon_category_series(base_path, filename, n_products=12, max_rows=300000):
    path = _resolve_amazon_path(base_path, filename)

    asin_counts = Counter()
    rows = []
    for row in _iter_jsonl(path, max_rows=max_rows):
        asin = row.get("parent_asin") or row.get("asin")
        ts = row.get("timestamp")
        if asin is None or ts is None:
            continue
        asin_counts[asin] += 1
        rows.append((asin, int(ts)))

    selected = [asin for asin, _ in asin_counts.most_common(n_products)]
    if not selected:
        return {}

    selected_set = set(selected)
    daily_map = {asin: defaultdict(float) for asin in selected}
    for asin, ts in rows:
        if asin not in selected_set:
            continue
        dt = pd.to_datetime(ts, unit="ms", utc=True).tz_convert(None).normalize()
        daily_map[asin][dt] += 1.0

    series_map = {}
    for asin in selected:
        daily = daily_map.get(asin, {})
        if not daily:
            continue
        df = pd.DataFrame({"date": list(daily.keys()), "sales": list(daily.values())}).sort_values("date")
        full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
        df["sales"] = df["sales"].fillna(0.0).astype(float)
        series_map[asin] = df[["date", "sales"]]

    return series_map


def main():
    parser = argparse.ArgumentParser(description="Exploratory analysis of demand regimes (10-15 products).")
    parser.add_argument("--datasets", type=str, default="m5,amazon")
    parser.add_argument("--n-products", type=int, default=12)
    parser.add_argument("--m5-cat", type=str, default="HOUSEHOLD")
    parser.add_argument("--favorita-family", type=str, default="CLEANING")
    parser.add_argument("--favorita-store", type=int, default=1)
    parser.add_argument("--amazon-file", type=str, default="Health_and_Household.jsonl.gz")
    parser.add_argument("--amazon-max-rows", type=int, default=300000)
    parser.add_argument("--out-dir", type=str, default="reports/regime_eda")
    args = parser.parse_args()

    selected = {s.strip().lower() for s in args.datasets.split(",") if s.strip()}

    os.makedirs(args.out_dir, exist_ok=True)
    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    all_rows = []

    def process_series_map(dataset_name, series_map):
        for sid, df in series_map.items():
            y = df["sales"].astype(float).values
            m = _compute_regime_metrics(y)
            row = {"dataset": dataset_name, "series_id": sid}
            row.update(m)
            all_rows.append(row)

            plot_name = f"{_safe_name(dataset_name)}__{_safe_name(sid)}.png"
            plot_path = os.path.join(plot_dir, plot_name)
            title = f"{dataset_name} | {sid} | {m['regime_label']} ({m['sbc_class']})"
            _plot_series(df["date"], y, title, plot_path)

    if "m5" in selected:
        m5_map = load_m5_category_series(
            base_path="data/raw/m5",
            cat_id=args.m5_cat,
            n_products=args.n_products,
            seed=42,
        )
        process_series_map("M5_WALMART", m5_map)

    if "favorita" in selected:
        fav_map = load_favorita_family_series(
            base_path="data/raw/favorita",
            family=args.favorita_family,
            store_nbr=args.favorita_store,
            n_products=args.n_products,
        )
        process_series_map("FAVORITA", fav_map)

    if "amazon" in selected:
        amz_map = load_amazon_category_series(
            base_path="data/raw/amazon_2023/review_categories",
            filename=args.amazon_file,
            n_products=args.n_products,
            max_rows=args.amazon_max_rows,
        )
        process_series_map("AMAZON_2023", amz_map)

    if not all_rows:
        raise RuntimeError("No series processed. Check datasets and category files.")

    out = pd.DataFrame(all_rows).sort_values(["dataset", "regime_label", "series_id"]).reset_index(drop=True)
    out_path = os.path.join(args.out_dir, "regime_summary.csv")
    out.to_csv(out_path, index=False)

    count_path = os.path.join(args.out_dir, "regime_counts.csv")
    out.groupby(["dataset", "regime_label"], as_index=False).size().to_csv(count_path, index=False)

    print(f"Saved: {out_path}")
    print(f"Saved: {count_path}")
    print(f"Saved plots: {plot_dir}")
    print(out.groupby(['dataset', 'regime_label']).size())


if __name__ == "__main__":
    main()
