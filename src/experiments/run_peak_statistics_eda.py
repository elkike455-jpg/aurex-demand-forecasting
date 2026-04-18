import argparse
import gzip
import json
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


def _peak_metrics(y, prominence_ratio=0.2):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return {
            "n_days": 0,
            "n_peaks": 0,
            "peaks_per_365": np.nan,
            "avg_peak_height": np.nan,
            "max_peak_height": np.nan,
            "avg_interpeak_days": np.nan,
            "std_interpeak_days": np.nan,
            "peak_irregularity_cv": np.nan,
            "prominence_used": np.nan,
        }, np.array([], dtype=int)

    rng = float(np.max(y) - np.min(y))
    prominence = max(rng * prominence_ratio, 1e-8)
    peaks = _find_peaks_simple(y, prominence=prominence)

    n_peaks = int(len(peaks))
    peaks_per_365 = float(n_peaks / n * 365.0) if n > 0 else np.nan

    if n_peaks > 0:
        peak_vals = y[peaks]
        avg_peak_height = float(np.mean(peak_vals))
        max_peak_height = float(np.max(peak_vals))
    else:
        avg_peak_height = np.nan
        max_peak_height = np.nan

    if n_peaks >= 2:
        intervals = np.diff(peaks).astype(float)
        avg_interpeak_days = float(np.mean(intervals))
        std_interpeak_days = float(np.std(intervals))
        peak_irregularity_cv = (
            float(std_interpeak_days / avg_interpeak_days) if avg_interpeak_days > 1e-12 else np.nan
        )
    else:
        avg_interpeak_days = np.nan
        std_interpeak_days = np.nan
        peak_irregularity_cv = np.nan

    metrics = {
        "n_days": n,
        "n_peaks": n_peaks,
        "peaks_per_365": peaks_per_365,
        "avg_peak_height": avg_peak_height,
        "max_peak_height": max_peak_height,
        "avg_interpeak_days": avg_interpeak_days,
        "std_interpeak_days": std_interpeak_days,
        "peak_irregularity_cv": peak_irregularity_cv,
        "prominence_used": prominence,
    }
    return metrics, peaks


def _plot_series_with_peaks(df, peaks, title, out_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(11, 3.8))
    plt.plot(df["date"], df["sales"], linewidth=1.0, label="sales")
    if len(peaks) > 0:
        plt.scatter(df["date"].iloc[peaks], df["sales"].iloc[peaks], s=14, label="peaks")
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("sales")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def _iter_jsonl_gz(path, max_rows=None):
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


def _load_amazon_selected(amazon_base_path, amazon_file, series_ids, max_rows=100000):
    path = os.path.join(amazon_base_path, amazon_file)
    if not os.path.exists(path) and amazon_file.endswith(".jsonl.gz"):
        alt = os.path.join(amazon_base_path, amazon_file[:-3])
        if os.path.exists(alt):
            path = alt
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


def _plot_dataset_summary(df_stats, out_dir):
    import matplotlib.pyplot as plt

    ds = df_stats.copy()
    ds["dataset_short"] = ds["dataset"].replace(
        {"M5_WALMART": "M5", "FAVORITA": "Favorita", "AMAZON_2023": "Amazon"}
    )

    # Plot 1: peaks per 365 by series
    for dataset_name, sub in ds.groupby("dataset"):
        sub = sub.sort_values("n_peaks", ascending=False).head(15).copy()
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(sub)), sub["n_peaks"].values)
        plt.xticks(np.arange(len(sub)), sub["series_id"].values, rotation=75, fontsize=7)
        plt.title(f"{dataset_name}: Number of Peaks per Series")
        plt.ylabel("n_peaks")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{dataset_name}_n_peaks_bar.png"), dpi=140)
        plt.close()

    # Plot 2: dataset mean peak stats
    grouped = (
        ds.groupby("dataset_short", as_index=False)[["n_peaks", "peaks_per_365", "avg_interpeak_days"]]
        .mean(numeric_only=True)
    )
    x = np.arange(len(grouped))
    width = 0.25
    plt.figure(figsize=(9, 4))
    plt.bar(x - width, grouped["n_peaks"], width=width, label="mean n_peaks")
    plt.bar(x, grouped["peaks_per_365"], width=width, label="mean peaks_per_365")
    plt.bar(x + width, grouped["avg_interpeak_days"].fillna(0), width=width, label="mean interpeak_days")
    plt.xticks(x, grouped["dataset_short"])
    plt.title("Peak Statistics by Dataset (Means)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dataset_peak_stats_means.png"), dpi=140)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Peak statistics EDA for selected regime series.")
    parser.add_argument("--regime-summary", type=str, default="reports/regime_eda_final/regime_summary.csv")
    parser.add_argument("--amazon-file", type=str, default="Health_and_Household.jsonl.gz")
    parser.add_argument("--amazon-max-rows", type=int, default=100000)
    parser.add_argument("--prominence-ratio", type=float, default=0.2)
    parser.add_argument("--out-dir", type=str, default="reports/peak_eda_final")
    args = parser.parse_args()

    df_reg = pd.read_csv(args.regime_summary)
    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "plots_series")
    os.makedirs(plots_dir, exist_ok=True)

    datasets = {}
    for ds_name in df_reg["dataset"].unique():
        datasets[ds_name] = df_reg[df_reg["dataset"] == ds_name]["series_id"].tolist()

    all_series = {}
    if "M5_WALMART" in datasets:
        all_series["M5_WALMART"] = _load_m5_selected("data/raw/m5", datasets["M5_WALMART"])
    if "FAVORITA" in datasets:
        all_series["FAVORITA"] = _load_favorita_selected("data/raw/favorita", datasets["FAVORITA"], store_nbr=1)
    if "AMAZON_2023" in datasets:
        all_series["AMAZON_2023"] = _load_amazon_selected(
            "data/raw/amazon_2023/review_categories",
            args.amazon_file,
            datasets["AMAZON_2023"],
            max_rows=args.amazon_max_rows,
        )

    rows = []
    for dataset_name, series_map in all_series.items():
        for sid, df in series_map.items():
            y = df["sales"].astype(float).values
            metrics, peaks = _peak_metrics(y, prominence_ratio=args.prominence_ratio)
            row = {"dataset": dataset_name, "series_id": sid}
            row.update(metrics)
            rows.append(row)

            safe_sid = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in sid)
            out_plot = os.path.join(plots_dir, f"{dataset_name}__{safe_sid}__peaks.png")
            title = f"{dataset_name} | {sid} | peaks={metrics['n_peaks']} | p365={metrics['peaks_per_365']:.2f}"
            _plot_series_with_peaks(df, peaks, title, out_plot)

    if not rows:
        raise RuntimeError("No series processed for peak analysis.")

    df_stats = pd.DataFrame(rows).sort_values(["dataset", "n_peaks"], ascending=[True, False]).reset_index(drop=True)
    stats_path = os.path.join(args.out_dir, "peak_stats_summary.csv")
    df_stats.to_csv(stats_path, index=False)

    dataset_summary = (
        df_stats.groupby("dataset", as_index=False)[
            ["n_peaks", "peaks_per_365", "avg_peak_height", "max_peak_height", "avg_interpeak_days", "peak_irregularity_cv"]
        ]
        .mean(numeric_only=True)
        .round(4)
    )
    dataset_path = os.path.join(args.out_dir, "peak_stats_dataset_means.csv")
    dataset_summary.to_csv(dataset_path, index=False)

    _plot_dataset_summary(df_stats, args.out_dir)

    print(f"Saved: {stats_path}")
    print(f"Saved: {dataset_path}")
    print(f"Saved plots (series): {plots_dir}")
    print(f"Saved plots (summary): {args.out_dir}")
    print(dataset_summary)


if __name__ == "__main__":
    main()
