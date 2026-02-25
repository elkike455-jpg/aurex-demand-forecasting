import pandas as pd
from data.database.db_connection import get_connection

def load_aurex(product_id: int, start_date=None, end_date=None) -> pd.DataFrame:
    """
    Returns a daily time series for a given product_id from your Postgres DB.
    Output columns: date, sales, price
    """

    query = """
        SELECT sale_date::date AS date,
               units_sold      AS sales,
               unit_price      AS price
        FROM daily_sales
        WHERE product_id = %s
    """
    params = [product_id]

    if start_date:
        query += " AND sale_date >= %s"
        params.append(start_date)
    if end_date:
        query += " AND sale_date <= %s"
        params.append(end_date)

    query += " ORDER BY sale_date;"

    conn = get_connection()
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    # ensure daily continuity (fill missing days with 0 sales)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_idx).rename_axis("date").reset_index()

    df["sales"] = df["sales"].fillna(0.0)
    df["price"] = df["price"].ffill().bfill()

    return df[["date", "sales", "price"]]