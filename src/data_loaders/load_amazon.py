import gzip
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


def _resolve_amazon_path(base_path, filename):
    """Resolve either .jsonl.gz or plain .json files in the configured folder."""
    candidates = []

    direct = os.path.join(base_path, filename)
    candidates.append(direct)

    if filename.endswith(".jsonl.gz"):
        candidates.append(os.path.join(base_path, filename[:-3]))
    elif filename.endswith(".jsonl"):
        candidates.append(os.path.join(base_path, filename + ".gz"))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Missing Amazon review file. Tried: {', '.join(candidates)}"
    )


def _iter_jsonl(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def load_amazon_series(
    base_path="data/raw/amazon_2023/review_categories",
    filename="Cell_Phones_and_Accessories.jsonl.gz",
    top_rank=1,
    max_rows=1_500_000,
) -> pd.DataFrame:
    """
    Build one daily demand-like series from Amazon reviews by selecting one
    high-activity ASIN and counting reviews per day as sales proxy.
    Output columns: date, sales, price
    """
    path = _resolve_amazon_path(base_path, filename)

    asin_counts = Counter()
    sampled_rows = []
    for i, row in enumerate(_iter_jsonl(path), start=1):
        asin = row.get("parent_asin") or row.get("asin")
        ts = row.get("timestamp")
        if asin is None or ts is None:
            continue
        asin_counts[asin] += 1
        sampled_rows.append((asin, ts))
        if max_rows is not None and i >= int(max_rows):
            break

    if not asin_counts:
        return pd.DataFrame(columns=["date", "sales", "price"])

    ranked = asin_counts.most_common(max(top_rank, 1))
    selected_asin = ranked[min(top_rank, len(ranked)) - 1][0]

    daily_counts = defaultdict(int)
    for asin, ts in sampled_rows:
        if asin != selected_asin:
            continue
        # Amazon 2023 timestamps are epoch milliseconds.
        dt = pd.to_datetime(int(ts), unit="ms", utc=True).tz_convert(None).normalize()
        daily_counts[dt] += 1

    if not daily_counts:
        return pd.DataFrame(columns=["date", "sales", "price"])

    daily = (
        pd.DataFrame({"date": list(daily_counts.keys()), "sales": list(daily_counts.values())})
        .sort_values("date")
        .reset_index(drop=True)
    )

    full_idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_idx).rename_axis("date").reset_index()
    daily["sales"] = daily["sales"].fillna(0.0).astype(float)
    # Price is not available in review file; keep NaN so feature pipeline fills with 0.
    daily["price"] = np.nan

    return daily[["date", "sales", "price"]]
