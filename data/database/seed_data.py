import random
import numpy as np
import pandas as pd
from datetime import date

from db_connection import get_connection


# --- CONFIG ---
START_DATE = "2016-01-01"   # 10 años hasta 2025-12-31 (aprox)
END_DATE   = "2025-12-31"

random.seed(42)
np.random.seed(42)


# Seasonal events (month, day) → affects seasonal products
EVENTS = {
    "christmas": ("12-15", "12-31", 3.5),   # boost factor
    "halloween": ("10-20", "10-31", 3.0),
    "valentine": ("02-10", "02-14", 2.8),
    "easter":    ("03-20", "04-20", 2.0),   # flexible window
    "canada_day":("06-25", "07-03", 2.5),
    "mothers":   ("05-05", "05-12", 2.3),
    "nhl_playoffs": ("04-10", "06-30", 1.8),
}


def in_window(d: pd.Timestamp, start_mmdd: str, end_mmdd: str) -> bool:
    """Check if date d is within a month-day window in the same year."""
    y = d.year
    start = pd.Timestamp(f"{y}-{start_mmdd}")
    end = pd.Timestamp(f"{y}-{end_mmdd}")
    return start <= d <= end


def get_products(conn):
    df = pd.read_sql("SELECT product_id, product_name, base_price, is_seasonal FROM products ORDER BY product_id;", conn)
    return df


def simulate_units(d: pd.Timestamp, is_seasonal: bool) -> int:
    """
    Generate realistic daily units:
    - stable products: baseline + weekly pattern + noise
    - seasonal products: low baseline most of the year + event spikes
    """
    dow = d.dayofweek  # Mon=0..Sun=6

    # Weekly pattern: weekends slightly higher
    weekly = 1.0 + (0.15 if dow in [5, 6] else 0.0)

    if not is_seasonal:
        baseline = 12  # stable baseline
        trend = (d.year - 2016) * 0.3  # tiny upward trend
        noise = np.random.normal(0, 3)
        units = baseline * weekly + trend + noise
        return max(0, int(round(units)))

    # Seasonal product baseline
    baseline = 2
    units = baseline * weekly + np.random.normal(0, 1.2)

    # Add event spikes
    spike_factor = 1.0
    for _, (s, e, factor) in EVENTS.items():
        if in_window(d, s, e):
            spike_factor = max(spike_factor, factor)

    units *= spike_factor
    return max(0, int(round(units)))


def simulate_price(base_price: float, is_seasonal: bool, units: int) -> float:
    """Price with small random variation; allow discounts when units are high."""
    variation = np.random.normal(0, 0.03)  # ~3% noise
    price = base_price * (1 + variation)

    # Discount effect during high-demand spikes (simulate promo competition)
    if is_seasonal and units > 15:
        price *= 0.92  # ~8% discount
    if not is_seasonal and units > 20:
        price *= 0.95

    return max(0.5, round(price, 2))


def main():
    conn = get_connection()

    products = get_products(conn)
    dates = pd.date_range(START_DATE, END_DATE, freq="D")

    rows = []
    for _, p in products.iterrows():
        pid = int(p["product_id"])
        base_price = float(p["base_price"])
        seasonal = bool(p["is_seasonal"])

        for d in dates:
            units = simulate_units(d, seasonal)
            price = simulate_price(base_price, seasonal, units)
            revenue = round(units * price, 2)

            rows.append((pid, d.date(), units, price, revenue))

    print(f"Generated rows: {len(rows):,}")

    # Insert efficiently
    cur = conn.cursor()
    cur.execute("TRUNCATE TABLE daily_sales;")  # clean previous runs

    insert_sql = """
        INSERT INTO daily_sales (product_id, sale_date, units_sold, unit_price, total_revenue)
        VALUES (%s, %s, %s, %s, %s)
    """
    cur.executemany(insert_sql, rows)
    conn.commit()

    cur.close()
    conn.close()
    print("✅ daily_sales filled successfully!")


if __name__ == "__main__":
    main()