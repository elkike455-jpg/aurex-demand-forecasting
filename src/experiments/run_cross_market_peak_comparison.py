import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def _find_peaks_simple(y, prominence):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return np.array([], dtype=int)

    peaks = []
    for i in range(1, len(y) - 1):
        if y[i] <= y[i - 1] or y[i] <= y[i + 1]:
            continue
        left_base = np.min(y[: i + 1])
        right_base = np.min(y[i:])
        prom = y[i] - max(left_base, right_base)
        if prom >= prominence:
            peaks.append(i)
    return np.asarray(peaks, dtype=int)


def _minmax_scale(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _load_m5_selected(m5_base_path, series_ids):
    sales_candidates = [
        os.path.join(m5_base_path, "sales_train_validation.csv"),
        os.path.join(m5_base_path, "sales_train_evaluation.csv"),
    ]
    sales_path = next((p for p in sales_candidates if os.path.exists(p)), None)
    cal_path = os.path.join(m5_base_path, "calendar.csv")
    if sales_path is None or not os.path.exists(cal_path):
        raise FileNotFoundError("Missing M5 files.")

    sales = pd.read_csv(sales_path)
    d_cols = [c for c in sales.columns if c.startswith("d_")]
    cal = pd.read_csv(cal_path, usecols=["d", "date"])
    d_to_date = pd.Series(pd.to_datetime(cal["date"]).values, index=cal["d"]).to_dict()

    series_map = {}
    sub = sales[sales["id"].isin(series_ids)].copy()
    for _, row in sub.iterrows():
        sid = row["id"]
        y = row[d_cols].astype(float).values
        df = pd.DataFrame({"date": [d_to_date[d] for d in d_cols], "sales": y})
        series_map[sid] = df.sort_values("date").reset_index(drop=True)
    return series_map


def _load_favorita_selected(favorita_base_path, series_ids, store_nbr=1, chunksize=1_000_000):
    train_path = os.path.join(favorita_base_path, "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Missing {train_path}")

    item_ids = []
    for sid in series_ids:
        if sid.startswith("item_"):
            try:
                item_ids.append(int(sid.replace("item_", "")))
            except ValueError:
                pass
    selected_items = set(item_ids)
    if not selected_items:
        return {}

    usecols = ["date", "store_nbr", "item_nbr", "unit_sales"]
    dtypes = {"store_nbr": "int16", "item_nbr": "int32", "unit_sales": "float32"}

    item_daily = defaultdict(lambda: defaultdict(float))
    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["date"],
        chunksize=chunksize,
        low_memory=True,
    ):
        chunk = chunk[(chunk["store_nbr"] == store_nbr) & (chunk["item_nbr"].isin(selected_items))]
        if chunk.empty:
            continue
        chunk["unit_sales"] = chunk["unit_sales"].clip(lower=0)
        grouped = chunk.groupby(["item_nbr", "date"], as_index=False)["unit_sales"].sum()
        for _, row in grouped.iterrows():
            key = f"item_{int(row['item_nbr'])}"
            item_daily[key][pd.Timestamp(row["date"]).normalize()] += float(row["unit_sales"])

    series_map = {}
    for sid in series_ids:
        daily = item_daily.get(sid, {})
        if not daily:
            continue
        df = pd.DataFrame({"date": list(daily.keys()), "sales": list(daily.values())}).sort_values("date")
        full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
        df["sales"] = df["sales"].fillna(0.0)
        series_map[sid] = df[["date", "sales"]]
    return series_map


def _iter_jsonl_gz(path, max_rows=None):
    import gzip
    import json

    with gzip.open(path, "rt", encoding="utf-8") as f:
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


def _load_amazon_selected(amazon_base_path, amazon_file, series_ids, max_rows=100000):
    path = os.path.join(amazon_base_path, amazon_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    selected = set(series_ids)
    daily_map = {sid: defaultdict(float) for sid in selected}
    for row in _iter_jsonl_gz(path, max_rows=max_rows):
        asin = row.get("parent_asin") or row.get("asin")
        ts = row.get("timestamp")
        if asin not in selected or ts is None:
            continue
        dt = pd.to_datetime(int(ts), unit="ms", utc=True).tz_convert(None).normalize()
        daily_map[asin][dt] += 1.0

    series_map = {}
    for sid in series_ids:
        daily = daily_map.get(sid, {})
        if not daily:
            continue
        df = pd.DataFrame({"date": list(daily.keys()), "sales": list(daily.values())}).sort_values("date")
        full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
        df["sales"] = df["sales"].fillna(0.0)
        series_map[sid] = df[["date", "sales"]]
    return series_map


def _build_triplets(merged):
    # These names are inferred labels to make cross-market comparison readable.
    anchors = [
        ("Paper roll (proxy)", "B0C1G1BJ2B"),
        ("Soap/Cleaner (proxy)", "B072KDRZH5"),
        ("Personal care (proxy)", "B09KC8ZYYX"),
    ]
    feat_cols = [
        "n_peaks",
        "peaks_per_365",
        "avg_peak_height",
        "max_peak_height",
        "avg_interpeak_days",
        "peak_irregularity_cv",
        "zero_rate",
        "adi",
        "cv2",
        "transition_score",
    ]
    work = merged.copy()
    work[feat_cols] = work[feat_cols].replace([np.inf, -np.inf], np.nan)
    work[feat_cols] = work[feat_cols].fillna(work[feat_cols].median(numeric_only=True))
    std = work[feat_cols].std().replace(0, 1.0)
    z = (work[feat_cols] - work[feat_cols].mean()) / std
    for c in feat_cols:
        work[f"z_{c}"] = z[c]

    used_m5 = set()
    used_fav = set()
    out = []

    for inferred_name, amz_sid in anchors:
        amz_row = work[(work["dataset"] == "AMAZON_2023") & (work["series_id"] == amz_sid)]
        if amz_row.empty:
            continue
        amz_row = amz_row.iloc[0]
        z_cols = [f"z_{c}" for c in feat_cols]
        amz_vec = amz_row[z_cols].values.astype(float)

        m5_cands = work[(work["dataset"] == "M5_WALMART") & (~work["series_id"].isin(used_m5))].copy()
        fav_cands = work[(work["dataset"] == "FAVORITA") & (~work["series_id"].isin(used_fav))].copy()
        if m5_cands.empty or fav_cands.empty:
            continue

        m5_dist = np.linalg.norm(m5_cands[z_cols].values - amz_vec, axis=1)
        fav_dist = np.linalg.norm(fav_cands[z_cols].values - amz_vec, axis=1)
        m5_sid = m5_cands.iloc[int(np.argmin(m5_dist))]["series_id"]
        fav_sid = fav_cands.iloc[int(np.argmin(fav_dist))]["series_id"]

        used_m5.add(m5_sid)
        used_fav.add(fav_sid)

        out.append(
            {
                "inferred_product": inferred_name,
                "M5_WALMART": m5_sid,
                "FAVORITA": fav_sid,
                "AMAZON_2023": amz_sid,
            }
        )
    return pd.DataFrame(out)


def _plot_triplet(series_triplet, series_map_by_ds, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"M5_WALMART": "#1f77b4", "FAVORITA": "#2ca02c", "AMAZON_2023": "#d62728"}
    plt.figure(figsize=(12, 4.5))
    for ds in ["M5_WALMART", "FAVORITA", "AMAZON_2023"]:
        sid = series_triplet[ds]
        df = series_map_by_ds[ds][sid].copy()
        y = _minmax_scale(df["sales"].values)
        x = np.arange(len(y))
        prom = max(float(np.max(y) - np.min(y)) * 0.2, 1e-8)
        peaks = _find_peaks_simple(y, prominence=prom)
        plt.plot(x, y, color=colors[ds], linewidth=1.1, label=f"{ds}: {sid}")
        if len(peaks) > 0:
            plt.scatter(x[peaks], y[peaks], color=colors[ds], s=10, alpha=0.8)

    plt.title(f"Cross-market peak comparison: {series_triplet['inferred_product']}")
    plt.xlabel("Relative day index")
    plt.ylabel("Normalized demand proxy (0-1)")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Cross-market peak comparison for inferred similar products.")
    parser.add_argument("--peak-summary", type=str, default="reports/peak_eda_final/peak_stats_summary.csv")
    parser.add_argument("--regime-summary", type=str, default="reports/regime_eda_final/regime_summary.csv")
    parser.add_argument("--amazon-file", type=str, default="Health_and_Household.jsonl.gz")
    parser.add_argument("--amazon-max-rows", type=int, default=100000)
    parser.add_argument("--out-dir", type=str, default="reports/cross_market_peak_comparison")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    peak_df = pd.read_csv(args.peak_summary)
    reg_df = pd.read_csv(args.regime_summary)
    merged = peak_df.merge(
        reg_df[["dataset", "series_id", "regime_label", "sbc_class", "zero_rate", "adi", "cv2", "transition_score"]],
        on=["dataset", "series_id"],
        how="left",
    )

    triplets = _build_triplets(merged)
    if triplets.empty:
        raise RuntimeError("Could not build triplets.")

    m5_ids = triplets["M5_WALMART"].tolist()
    fav_ids = triplets["FAVORITA"].tolist()
    amz_ids = triplets["AMAZON_2023"].tolist()

    series_map_by_ds = {
        "M5_WALMART": _load_m5_selected("data/raw/m5", m5_ids),
        "FAVORITA": _load_favorita_selected("data/raw/favorita", fav_ids, store_nbr=1),
        "AMAZON_2023": _load_amazon_selected(
            "data/raw/amazon_2023/review_categories",
            args.amazon_file,
            amz_ids,
            max_rows=args.amazon_max_rows,
        ),
    }

    missing = []
    for ds in ["M5_WALMART", "FAVORITA", "AMAZON_2023"]:
        for sid in triplets[ds].tolist():
            if sid not in series_map_by_ds[ds]:
                missing.append((ds, sid))
    if missing:
        raise RuntimeError(f"Missing series data for: {missing}")

    for _, row in triplets.iterrows():
        safe = row["inferred_product"].lower().replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        out_plot = os.path.join(args.out_dir, f"{safe}.png")
        _plot_triplet(row.to_dict(), series_map_by_ds, out_plot)

    out_csv = os.path.join(args.out_dir, "inferred_triplets.csv")
    triplets.to_csv(out_csv, index=False)
    print(f"Saved triplets: {out_csv}")
    print(f"Saved plots in: {args.out_dir}")
    print(triplets)


if __name__ == "__main__":
    main()
