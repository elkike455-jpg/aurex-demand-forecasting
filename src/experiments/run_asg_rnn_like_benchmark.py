from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.common_protocol import OFFICIAL_BENCHMARK_PROTOCOL, split_panel_protocol
from src.experiments.stat_benchmark_utils import (
    BENCHMARK_PRODUCTS,
    flat_nonflat_label,
    summarize_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
M5_BASE = REPO_ROOT / "data" / "raw" / "m5"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "asg_rnn_like_benchmark"
BENCHMARK_LABEL_BY_ID = dict(BENCHMARK_PRODUCTS)


@dataclass(frozen=True)
class ASGRNNLikeConfig:
    context_length: int = OFFICIAL_BENCHMARK_PROTOCOL.context_length
    hidden_size: int = 32
    static_hidden_size: int = 12
    graph_hidden_size: int = 32
    epochs: int = 18
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    top_k: int = 5
    expanded_products: int = 48
    seed: int = OFFICIAL_BENCHMARK_PROTOCOL.seed
    max_days: int = OFFICIAL_BENCHMARK_PROTOCOL.max_days
    val_days: int = OFFICIAL_BENCHMARK_PROTOCOL.val_days
    test_days: int = OFFICIAL_BENCHMARK_PROTOCOL.test_days


def _set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _sales_path() -> Path:
    for name in ("sales_train_validation.csv", "sales_train_evaluation.csv"):
        path = M5_BASE / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing M5 sales file under {M5_BASE}")


def _load_calendar(max_days: int) -> pd.DataFrame:
    calendar = pd.read_csv(M5_BASE / "calendar.csv")
    calendar["date"] = pd.to_datetime(calendar["date"])
    return calendar.tail(int(max_days)).reset_index(drop=True)


def _profile_series(y: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    nonzero = y[y > 0]
    mean_sales = float(np.mean(y))
    std_sales = float(np.std(y))
    zero_rate = float(np.mean(y == 0))
    nonzero_days = int(np.sum(y > 0))
    max_sales = float(np.max(y)) if len(y) else 0.0
    cv = float(std_sales / mean_sales) if mean_sales > 0 else 0.0
    adi = float(len(y) / nonzero_days) if nonzero_days > 0 else float(len(y))
    nonzero_mean = float(np.mean(nonzero)) if nonzero_days else 0.0
    nonzero_std = float(np.std(nonzero)) if nonzero_days else 0.0
    cv2 = float((nonzero_std / nonzero_mean) ** 2) if nonzero_mean > 0 else 0.0
    return {
        "mean_sales": mean_sales,
        "std_sales": std_sales,
        "zero_rate": zero_rate,
        "nonzero_days": float(nonzero_days),
        "max_sales": max_sales,
        "cv": cv,
        "ADI": adi,
        "CV2": cv2,
    }


def _classify_profile(profile: Dict[str, float]) -> str:
    zero_rate = float(profile["zero_rate"])
    mean_sales = float(profile["mean_sales"])
    nonzero_days = float(profile["nonzero_days"])
    if zero_rate <= 0.20 and mean_sales >= 1.0:
        return "stable_or_high"
    if zero_rate >= 0.80:
        return "low_volume"
    if 0.35 <= zero_rate <= 0.80 and nonzero_days >= 28:
        return "intermittent"
    return "other"


def load_sales_metadata(max_days: int, state_id: str = "CA") -> Tuple[pd.DataFrame, List[str]]:
    sales_path = _sales_path()
    header = pd.read_csv(sales_path, nrows=0).columns.tolist()
    d_cols = [c for c in header if c.startswith("d_")][-int(max_days) :]
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_df = pd.read_csv(sales_path, usecols=id_cols + d_cols)
    sales_df = sales_df.loc[sales_df["state_id"] == state_id].copy().reset_index(drop=True)
    profiles = []
    values = sales_df[d_cols].to_numpy(dtype=float)
    for idx in range(len(sales_df)):
        profile = _profile_series(values[idx])
        profile["demand_class"] = _classify_profile(profile)
        profiles.append(profile)
    return pd.concat([sales_df[id_cols], pd.DataFrame(profiles), sales_df[d_cols]], axis=1), d_cols


def select_official_16_ids(profile_df: pd.DataFrame) -> List[str]:
    path = REPO_ROOT / "reports" / "gnn_benchmarks" / "correlation_matrix_16_products" / "profile_features_16_products.csv"
    if path.exists():
        ids = pd.read_csv(path)["id"].astype(str).tolist()
        for series_id, _ in BENCHMARK_PRODUCTS:
            if series_id not in ids:
                ids.append(series_id)
        return ids
    return select_expanded_ids(profile_df, 16, OFFICIAL_BENCHMARK_PROTOCOL.seed)


def select_expanded_ids(profile_df: pd.DataFrame, total_products: int, seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    selected = [series_id for series_id, _ in BENCHMARK_PRODUCTS]
    per_class = max(1, total_products // 3)
    for demand_class in ("stable_or_high", "intermittent", "low_volume"):
        existing = int((profile_df.loc[profile_df["id"].isin(selected), "demand_class"] == demand_class).sum())
        need = max(0, per_class - existing)
        candidates = profile_df.loc[
            (profile_df["demand_class"] == demand_class) & (~profile_df["id"].isin(selected))
        ].copy()
        if demand_class == "low_volume":
            candidates = candidates.sort_values(["zero_rate", "nonzero_days", "id"], ascending=[False, True, True])
        else:
            candidates = candidates.sort_values(["mean_sales", "nonzero_days", "id"], ascending=[False, False, True])
        if need:
            pool = candidates.head(max(need * 4, need))
            selected.extend(rng.choice(pool["id"].to_numpy(), size=min(need, len(pool)), replace=False).tolist())
    if len(selected) < total_products:
        rest = profile_df.loc[~profile_df["id"].isin(selected)].sort_values(
            ["nonzero_days", "mean_sales", "id"], ascending=[False, False, True]
        )
        selected.extend(rest["id"].head(total_products - len(selected)).tolist())
    return selected[:total_products]


def load_panel_for_ids(ids: Iterable[str], max_days: int) -> Dict[str, object]:
    ids = list(dict.fromkeys(ids))
    sales_df, d_cols = load_sales_metadata(max_days=max_days, state_id=OFFICIAL_BENCHMARK_PROTOCOL.state_id)
    work = sales_df.loc[sales_df["id"].isin(ids)].copy()
    missing = sorted(set(ids) - set(work["id"]))
    if missing:
        raise ValueError(f"Missing requested M5 series ids: {missing}")
    order = {series_id: idx for idx, series_id in enumerate(ids)}
    work["__order"] = work["id"].map(order)
    work = work.sort_values("__order").drop(columns=["__order"]).reset_index(drop=True)

    calendar = _load_calendar(max_days=max_days)
    d_cols = [d for d in calendar["d"].tolist() if d in d_cols]
    sales = work[d_cols].to_numpy(dtype=np.float32)

    prices = pd.read_csv(M5_BASE / "sell_prices.csv")
    price_panel = []
    for _, row in work.iterrows():
        ts = pd.DataFrame({"d": d_cols}).merge(calendar[["d", "wm_yr_wk"]], on="d", how="left")
        price_sub = prices.loc[
            (prices["store_id"] == row["store_id"]) & (prices["item_id"] == row["item_id"]),
            ["wm_yr_wk", "sell_price"],
        ].drop_duplicates("wm_yr_wk")
        ts = ts.merge(price_sub, on="wm_yr_wk", how="left")
        ts["sell_price"] = ts["sell_price"].ffill().bfill().fillna(0.0)
        price_panel.append(ts["sell_price"].to_numpy(dtype=np.float32))

    metadata_cols = [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "mean_sales",
        "std_sales",
        "zero_rate",
        "nonzero_days",
        "max_sales",
        "cv",
        "ADI",
        "CV2",
        "demand_class",
    ]
    return {
        "sales": sales,
        "prices": np.asarray(price_panel, dtype=np.float32),
        "dates": pd.DatetimeIndex(calendar["date"]),
        "metadata": work[metadata_cols].copy().reset_index(drop=True),
    }


def _positive_corr(values: np.ndarray) -> np.ndarray:
    if values.shape[1] < 3:
        return np.eye(values.shape[0], dtype=np.float32)
    corr = np.corrcoef(values)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr.astype(np.float32)


def build_hybrid_graph(metadata: pd.DataFrame, train_sales: np.ndarray, train_prices: np.ndarray, top_k: int):
    n = len(metadata)
    sales_sim = _positive_corr(np.log1p(np.asarray(train_sales, dtype=float)))
    price_sim = _positive_corr(np.asarray(train_prices, dtype=float))
    meta_sim = np.zeros((n, n), dtype=np.float32)
    for i, left in metadata.iterrows():
        for j, right in metadata.iterrows():
            if i == j:
                meta_sim[i, j] = 1.0
                continue
            score = 0.0
            score += 1.0 if left["cat_id"] == right["cat_id"] else 0.0
            score += 1.5 if left["dept_id"] == right["dept_id"] else 0.0
            score += 1.0 if left["store_id"] == right["store_id"] else 0.0
            score += 0.5 if left["state_id"] == right["state_id"] else 0.0
            meta_sim[i, j] = score / 4.0

    profile_cols = ["mean_sales", "std_sales", "zero_rate", "nonzero_days", "max_sales", "ADI", "CV2"]
    profile = metadata[profile_cols].astype(float)
    spans = (profile.max(axis=0) - profile.min(axis=0)).replace(0.0, 1.0)
    scaled = ((profile - profile.min(axis=0)) / spans).to_numpy(dtype=float)
    dist = np.sqrt(((scaled[:, None, :] - scaled[None, :, :]) ** 2).sum(axis=2))
    profile_sim = np.clip(1.0 - dist / np.sqrt(len(profile_cols)), 0.0, 1.0)
    np.fill_diagonal(profile_sim, 1.0)

    hybrid = 0.35 * sales_sim + 0.20 * price_sim + 0.30 * meta_sim + 0.15 * profile_sim
    np.fill_diagonal(hybrid, 1.0)
    adj = np.eye(n, dtype=np.float32)
    ids = metadata["id"].astype(str).tolist()
    edge_rows = []
    for i in range(n):
        scores = hybrid[i].copy()
        scores[i] = -1.0
        for j in np.argsort(scores)[-int(top_k) :]:
            if scores[j] > 0:
                adj[i, j] = float(scores[j])
                edge_rows.append(
                    {
                        "source_product": ids[i],
                        "target_product": ids[j],
                        "hybrid_similarity": float(scores[j]),
                        "sales_similarity": float(sales_sim[i, j]),
                        "price_similarity": float(price_sim[i, j]),
                        "metadata_similarity": float(meta_sim[i, j]),
                        "profile_similarity": float(profile_sim[i, j]),
                    }
                )
    adj = np.maximum(adj, adj.T)
    degree = adj.sum(axis=1, keepdims=True)
    degree[degree <= 0] = 1.0
    sim_df = pd.DataFrame(hybrid, index=ids, columns=ids)
    edge_df = pd.DataFrame(edge_rows).drop_duplicates(["source_product", "target_product"])
    return (adj / degree).astype(np.float32), sim_df, edge_df


def build_static_features(metadata: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    pieces = []
    names = []
    for col in ["cat_id", "dept_id", "store_id", "state_id", "demand_class"]:
        dummies = pd.get_dummies(metadata[col].astype(str), prefix=col)
        pieces.append(dummies)
        names.extend(dummies.columns.tolist())
    numeric_cols = ["mean_sales", "std_sales", "zero_rate", "nonzero_days", "max_sales", "ADI", "CV2"]
    nums = metadata[numeric_cols].astype(float)
    nums = (nums - nums.mean(axis=0)) / nums.std(axis=0).replace(0.0, 1.0)
    nums = nums.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    pieces.append(nums)
    names.extend(numeric_cols)
    return pd.concat(pieces, axis=1).to_numpy(dtype=np.float32), names


def _node_scales(y: np.ndarray) -> np.ndarray:
    scales = []
    for row in np.asarray(y, dtype=float):
        nz = np.abs(row[row != 0])
        scales.append(max(float(np.mean(nz)) if nz.size else 1.0, 1.0))
    return np.asarray(scales, dtype=np.float32)


def _price_norm(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(p, axis=1).astype(np.float32)
    std = np.std(p, axis=1).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def _build_windows(sales_scaled: np.ndarray, prices_scaled: np.ndarray, context_length: int):
    n_nodes, total_steps = sales_scaled.shape
    samples = total_steps - int(context_length)
    x = np.zeros((samples, n_nodes, int(context_length), 2), dtype=np.float32)
    y = np.zeros((samples, n_nodes), dtype=np.float32)
    for idx in range(samples):
        end = idx + int(context_length)
        x[idx, :, :, 0] = sales_scaled[:, idx:end]
        x[idx, :, :, 1] = prices_scaled[:, idx:end]
        y[idx] = sales_scaled[:, end]
    return x, y


class ASGRNNLikeNet(nn.Module):
    def __init__(self, dynamic_features: int, static_features: int, config: ASGRNNLikeConfig):
        super().__init__()
        self.temporal = nn.GRU(dynamic_features, config.hidden_size, batch_first=True)
        self.static_encoder = nn.Sequential(nn.Linear(static_features, config.static_hidden_size), nn.ReLU())
        fused = config.hidden_size + config.static_hidden_size
        self.graph_layer = nn.Sequential(nn.Linear(fused, config.graph_hidden_size), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(fused + config.graph_hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, 1),
        )

    def forward(self, x: torch.Tensor, static_x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, context, n_features = x.shape
        encoded, _ = self.temporal(x.reshape(batch * n_nodes, context, n_features))
        temporal = encoded[:, -1, :].reshape(batch, n_nodes, -1)
        static = self.static_encoder(static_x).unsqueeze(0).expand(batch, -1, -1)
        local = torch.cat([temporal, static], dim=-1)
        graph = self.graph_layer(torch.matmul(adj.unsqueeze(0), local))
        return self.head(torch.cat([local, graph], dim=-1)).squeeze(-1)


class ASGRNNLikeForecaster:
    def __init__(self, config: ASGRNNLikeConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: ASGRNNLikeNet | None = None
        self.sales_scale_: np.ndarray | None = None
        self.price_mean_: np.ndarray | None = None
        self.price_std_: np.ndarray | None = None
        self.final_loss_: float = float("nan")

    def fit(self, sales: np.ndarray, prices: np.ndarray, static_x: np.ndarray, adj: np.ndarray):
        _set_seeds(self.config.seed)
        self.sales_scale_ = _node_scales(sales)
        self.price_mean_, self.price_std_ = _price_norm(prices)
        sales_scaled = (sales / self.sales_scale_[:, None]).astype(np.float32)
        prices_scaled = ((prices - self.price_mean_[:, None]) / self.price_std_[:, None]).astype(np.float32)
        x, y = _build_windows(sales_scaled, prices_scaled, self.config.context_length)
        self.model = ASGRNNLikeNet(x.shape[-1], static_x.shape[-1], self.config).to(self.device)
        static_tensor = torch.from_numpy(static_x.astype(np.float32)).to(self.device)
        adj_tensor = torch.from_numpy(adj.astype(np.float32)).to(self.device)
        loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=self.config.batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        for _ in range(self.config.epochs):
            losses = []
            for bx, by in loader:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                pred = self.model(bx, static_tensor, adj_tensor)
                loss = nn.functional.huber_loss(pred, by, delta=1.0)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
            self.final_loss_ = float(np.mean(losses)) if losses else float("nan")
        return self

    def forecast(self, h: int, history_sales: np.ndarray, history_prices: np.ndarray, future_prices: np.ndarray, static_x: np.ndarray, adj: np.ndarray) -> np.ndarray:
        if self.model is None or self.sales_scale_ is None or self.price_mean_ is None or self.price_std_ is None:
            raise RuntimeError("Model is not fitted.")
        sales_work = (history_sales / self.sales_scale_[:, None]).astype(np.float32)
        price_work = ((history_prices - self.price_mean_[:, None]) / self.price_std_[:, None]).astype(np.float32)
        future_price_scaled = ((future_prices - self.price_mean_[:, None]) / self.price_std_[:, None]).astype(np.float32)
        static_tensor = torch.from_numpy(static_x.astype(np.float32)).to(self.device)
        adj_tensor = torch.from_numpy(adj.astype(np.float32)).to(self.device)
        preds = np.zeros((sales_work.shape[0], int(h)), dtype=np.float32)
        self.model.eval()
        with torch.no_grad():
            for step in range(int(h)):
                x = np.stack(
                    [sales_work[:, -self.config.context_length :], price_work[:, -self.config.context_length :]],
                    axis=-1,
                )[None, :, :, :]
                pred = self.model(torch.from_numpy(x).to(self.device), static_tensor, adj_tensor).squeeze(0)
                next_sales = pred.detach().cpu().numpy().astype(np.float32)
                preds[:, step] = next_sales
                sales_work = np.concatenate([sales_work, next_sales[:, None]], axis=1)
                price_work = np.concatenate([price_work, future_price_scaled[:, step : step + 1]], axis=1)
        return np.maximum(preds * self.sales_scale_[:, None], 0.0)


def _save_graph(edge_df: pd.DataFrame, metadata: pd.DataFrame, figure_path: Path) -> Dict[str, float]:
    graph = nx.Graph()
    ids = metadata["id"].astype(str).tolist()
    graph.add_nodes_from(ids)
    for row in edge_df.itertuples(index=False):
        graph.add_edge(row.source_product, row.target_product, weight=float(row.hybrid_similarity))
    components = list(nx.connected_components(graph))
    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42, weight="weight", k=0.8, iterations=250)
    cats = sorted(metadata["cat_id"].astype(str).unique())
    palette = plt.cm.get_cmap("tab10", max(len(cats), 1))
    color_by_cat = {cat: palette(i) for i, cat in enumerate(cats)}
    node_colors = [color_by_cat[metadata.loc[metadata["id"] == n, "cat_id"].iloc[0]] for n in graph.nodes()]
    edge_widths = [1.0 + 5.0 * graph[u][v].get("weight", 0.0) for u, v in graph.edges()]
    nx.draw_networkx_edges(graph, pos, width=edge_widths, alpha=0.45, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=430, edgecolors="white", linewidths=0.8, ax=ax)
    nx.draw_networkx_labels(graph, pos, labels={n: n[:18] for n in graph.nodes()}, font_size=7, ax=ax)
    ax.set_axis_off()
    ax.set_title("ASG-RNN-like hybrid product graph")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "nodes": float(graph.number_of_nodes()),
        "edges": float(graph.number_of_edges()),
        "density": float(nx.density(graph)) if graph.number_of_nodes() > 1 else 0.0,
        "connected_components": float(len(components)),
        "largest_component_size": float(max((len(c) for c in components), default=0)),
    }


def _plot_prediction(pred_df: pd.DataFrame, figure_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(pd.to_datetime(pred_df["date"]), pred_df["sales"], label="Real", linewidth=2)
    ax.plot(pd.to_datetime(pred_df["date"]), pred_df["y_pred"], label="Predicted", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_one_panel(run_name: str, ids: List[str], out_dir: Path, config: ASGRNNLikeConfig):
    panel = load_panel_for_ids(ids, max_days=config.max_days)
    sales = np.asarray(panel["sales"], dtype=np.float32)
    prices = np.asarray(panel["prices"], dtype=np.float32)
    metadata = pd.DataFrame(panel["metadata"]).copy().reset_index(drop=True)
    dates = pd.DatetimeIndex(panel["dates"])
    train_sales, val_sales, test_sales = split_panel_protocol(sales, val_days=config.val_days, test_days=config.test_days)
    train_prices, val_prices, test_prices = split_panel_protocol(prices, val_days=config.val_days, test_days=config.test_days)
    fit_sales = np.concatenate([train_sales, val_sales], axis=1)
    fit_prices = np.concatenate([train_prices, val_prices], axis=1)
    adj, sim_df, edge_df = build_hybrid_graph(metadata, fit_sales, fit_prices, top_k=config.top_k)
    static_x, static_names = build_static_features(metadata)
    model = ASGRNNLikeForecaster(config).fit(fit_sales, fit_prices, static_x, adj)
    pred_panel = model.forecast(test_sales.shape[1], fit_sales, fit_prices, test_prices, static_x, adj)

    graph_dir = out_dir / "graph_artifacts"
    metadata.to_csv(graph_dir / f"{run_name}_metadata.csv", index=False)
    sim_df.to_csv(graph_dir / f"{run_name}_hybrid_similarity.csv")
    edge_df.to_csv(graph_dir / f"{run_name}_graph_edges.csv", index=False)
    pd.DataFrame(adj, index=metadata["id"], columns=metadata["id"]).to_csv(graph_dir / f"{run_name}_adjacency.csv")
    graph_summary = _save_graph(edge_df, metadata, graph_dir / f"{run_name}_graph.png")
    graph_summary["run_name"] = run_name

    metrics_rows = []
    pred_frames = []
    for idx, meta in metadata.loc[metadata["id"].isin(BENCHMARK_LABEL_BY_ID)].iterrows():
        series_id = str(meta["id"])
        label = BENCHMARK_LABEL_BY_ID[series_id]
        y_true = test_sales[idx].astype(float)
        y_pred = pred_panel[idx].astype(float)
        metrics = summarize_predictions(y_true, y_pred)
        metrics_rows.append(
            {
                "run_name": run_name,
                "series_id": series_id,
                "benchmark_label": label,
                "model": "ASG_RNN_LIKE",
                "n_products": int(len(ids)),
                "max_days": config.max_days,
                "train_days": int(train_sales.shape[1]),
                "val_days": int(val_sales.shape[1]),
                "fit_days": int(fit_sales.shape[1]),
                "test_days": int(test_sales.shape[1]),
                "context_length": config.context_length,
                "final_loss": float(model.final_loss_),
                **metrics,
                "flat_nonflat": flat_nonflat_label(metrics["variance_ratio"]),
            }
        )
        pred_df = pd.DataFrame(
            {
                "date": dates[-test_sales.shape[1] :],
                "sales": y_true,
                "y_pred": y_pred,
                "series_id": series_id,
                "benchmark_label": label,
                "run_name": run_name,
            }
        )
        pred_df.to_csv(out_dir / "per_series" / f"{run_name}_{label}_{series_id}_predictions.csv", index=False)
        _plot_prediction(
            pred_df,
            out_dir / "figures" / f"{run_name}_{label}_{series_id}_real_vs_predicted.png",
            f"ASG-RNN-like {run_name}: {series_id}",
        )
        pred_frames.append(pred_df)

    train_row = {
        "run_name": run_name,
        "model": "ASG_RNN_LIKE",
        "n_products": int(len(ids)),
        "context_length": config.context_length,
        "epochs": config.epochs,
        "hidden_size": config.hidden_size,
        "top_k": config.top_k,
        "final_loss": float(model.final_loss_),
        "static_feature_count": int(static_x.shape[1]),
        "static_features": json.dumps(static_names),
    }
    return pd.DataFrame(metrics_rows), pd.concat(pred_frames, ignore_index=True), train_row, graph_summary


def run_experiment(out_dir: Path = DEFAULT_OUT_DIR, config: ASGRNNLikeConfig = ASGRNNLikeConfig()) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("figures", "per_series", "graph_artifacts"):
        (out_dir / subdir).mkdir(exist_ok=True)
    profile_df, _ = load_sales_metadata(max_days=config.max_days, state_id=OFFICIAL_BENCHMARK_PROTOCOL.state_id)
    official_ids = select_official_16_ids(profile_df)
    expanded_ids = select_expanded_ids(profile_df, config.expanded_products, config.seed)

    frames = []
    pred_frames = []
    training_rows = []
    graph_rows = []
    for run_name, ids in [("official_16", official_ids), ("expanded_48", expanded_ids)]:
        metrics_df, pred_df, train_row, graph_row = run_one_panel(run_name, ids, out_dir, config)
        frames.append(metrics_df)
        pred_frames.append(pred_df)
        training_rows.append(train_row)
        graph_rows.append(graph_row)

    metrics_df = pd.concat(frames, ignore_index=True)
    predictions_df = pd.concat(pred_frames, ignore_index=True)
    training_df = pd.DataFrame(training_rows)
    graph_df = pd.DataFrame(graph_rows)
    expanded_meta = pd.read_csv(out_dir / "graph_artifacts" / "expanded_48_metadata.csv")
    counts = expanded_meta["demand_class"].value_counts().rename_axis("demand_class").reset_index(name="count")

    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    predictions_df.to_csv(out_dir / "predictions.csv", index=False)
    training_df.to_csv(out_dir / "training_summary.csv", index=False)
    graph_df.to_csv(out_dir / "graph_summary.csv", index=False)
    counts.to_csv(out_dir / "expanded_panel_demand_class_counts.csv", index=False)
    (out_dir / "interpretation.md").write_text(
        "\n".join(
            [
                "# ASG-RNN-like Benchmark Run",
                "",
                "This is an ASG-RNN-like adaptation, not a verified reproduction of the ASG-RNN paper.",
                "Local search did not find a direct ASG-RNN implementation. The experiment uses the existing M5 protocol and an engineered hybrid graph.",
                "",
                f"Protocol: max_days={config.max_days}, context_length={config.context_length}, val_days={config.val_days}, test_days={config.test_days}, seed={config.seed}.",
                f"Benchmark products: {', '.join(BENCHMARK_LABEL_BY_ID.keys())}.",
                "Graph construction: top-k hybrid similarity using log-sales correlation, price correlation, M5 metadata similarity, and demand-profile similarity.",
                "Model: GRU temporal encoder + static node feature encoder + adjacency-weighted neighbor aggregation + point forecast head.",
                "Expanded panel rule: reproducible balanced sampling from CA M5 products across stable/high, intermittent, and low-volume demand classes, while forcing the benchmark trio to remain in the panel.",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "out_dir": str(out_dir),
        "metrics_path": str(out_dir / "metrics.csv"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "training_summary_path": str(out_dir / "training_summary.csv"),
        "graph_summary_path": str(out_dir / "graph_summary.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ASG-RNN-like M5 graph-recurrent benchmark.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=ASGRNNLikeConfig.epochs)
    parser.add_argument("--expanded-products", type=int, default=ASGRNNLikeConfig.expanded_products)
    parser.add_argument("--top-k", type=int, default=ASGRNNLikeConfig.top_k)
    parser.add_argument("--seed", type=int, default=ASGRNNLikeConfig.seed)
    args = parser.parse_args()
    config = ASGRNNLikeConfig(epochs=args.epochs, expanded_products=args.expanded_products, top_k=args.top_k, seed=args.seed)
    print(json.dumps(run_experiment(out_dir=args.out_dir, config=config), indent=2))


if __name__ == "__main__":
    main()
