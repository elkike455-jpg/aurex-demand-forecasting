from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_loaders.load_m5_panel import load_m5_panel_subset


@dataclass(frozen=True)
class CorrelationNotebookConfig:
    num_products: int = 16
    state_id: str = "CA"
    store_id: str | None = None
    cat_id: str | None = None
    dept_id: str | None = None
    max_days: int = 365
    min_nonzero_days: int = 28
    seed: int = 42
    corr_method: str = "pearson"
    reports_subdir: str = "correlation_matrix_16_products"


CONFIG = CorrelationNotebookConfig()


def resolve_m5_base(repo_root: Path | None = None) -> Path:
    repo_root = REPO_ROOT if repo_root is None else Path(repo_root)
    candidates = [
        repo_root / "data" / "raw" / "m5",
        Path.cwd() / "data" / "raw" / "m5",
    ]
    return next((path for path in candidates if path.exists()), candidates[0])


def load_gnn_subset(config: CorrelationNotebookConfig = CONFIG) -> dict[str, object]:
    m5_base = resolve_m5_base()
    panel = load_m5_panel_subset(
        base_path=str(m5_base),
        num_products=config.num_products,
        seed=config.seed,
        state_id=config.state_id,
        store_id=config.store_id,
        cat_id=config.cat_id,
        dept_id=config.dept_id,
        min_nonzero_days=config.min_nonzero_days,
        max_days=config.max_days,
    )
    return panel


def build_sales_frame(panel: dict[str, object]) -> pd.DataFrame:
    sales = np.asarray(panel["sales"], dtype=float)
    metadata = pd.DataFrame(panel["metadata"]).copy().reset_index(drop=True)
    dates = pd.DatetimeIndex(panel["dates"])
    sales_df = pd.DataFrame(sales.T, index=dates, columns=metadata["id"].astype(str).tolist())
    sales_df.index.name = "date"
    return sales_df


def compute_top_pairs(matrix_df: pd.DataFrame, value_col: str = "correlation") -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cols = matrix_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            value = float(matrix_df.iloc[i, j])
            rows.append(
                {
                    "product_a": cols[i],
                    "product_b": cols[j],
                    value_col: value,
                    f"abs_{value_col}": abs(value),
                }
            )
    return pd.DataFrame(rows).sort_values(
        [f"abs_{value_col}", value_col],
        ascending=[False, False],
    ).reset_index(drop=True)


def summarize_connectivity(
    metadata: pd.DataFrame,
    matrix_df: pd.DataFrame,
    value_col: str = "correlation",
) -> pd.DataFrame:
    labels = metadata[["id", "cat_id", "dept_id", "store_id"]].copy().reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for _, left in labels.iterrows():
        for _, right in labels.iterrows():
            rows.append(
                {
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "left_cat": left["cat_id"],
                    "right_cat": right["cat_id"],
                    "same_product": left["id"] == right["id"],
                    value_col: float(matrix_df.loc[left["id"], right["id"]]),
                }
            )
    pair_df = pd.DataFrame(rows)
    pair_df = pair_df.loc[~pair_df["same_product"]].copy()
    pair_df["pair_type"] = np.where(
        pair_df["left_cat"] == pair_df["right_cat"],
        "within_category",
        "cross_category",
    )

    summary = (
        pair_df.groupby(["left_cat", "pair_type"], as_index=False)[value_col]
        .mean()
        .rename(columns={"left_cat": "category", value_col: f"mean_{value_col}"})
        .sort_values(["category", "pair_type"])
        .reset_index(drop=True)
    )
    return summary


def threshold_edge_summary(corr_df: pd.DataFrame, thresholds: list[float] | None = None) -> pd.DataFrame:
    thresholds = [0.5, 0.8, 0.9, 0.99] if thresholds is None else thresholds
    top_pairs_df = compute_top_pairs(corr_df)
    rows = []
    total_pairs = int(len(top_pairs_df))
    for threshold in thresholds:
        mask = top_pairs_df["correlation"] >= float(threshold)
        edge_count = int(mask.sum())
        rows.append(
            {
                "threshold": float(threshold),
                "edge_count": edge_count,
                "pair_coverage": float(edge_count / max(total_pairs, 1)),
            }
        )
    return pd.DataFrame(rows)


def profile_series(series: np.ndarray) -> dict[str, float]:
    y = np.asarray(series, dtype=float)
    nonzero = y[y > 0]
    mean_sales = float(np.mean(y))
    std_sales = float(np.std(y, ddof=0))
    zero_rate = float(np.mean(y == 0))
    nonzero_days = int(np.sum(y > 0))
    max_sales = float(np.max(y))
    cv = float(std_sales / mean_sales) if mean_sales > 0 else 0.0
    adi = float(len(y) / nonzero_days) if nonzero_days > 0 else float(len(y))
    nonzero_mean = float(np.mean(nonzero)) if nonzero_days > 0 else 0.0
    nonzero_std = float(np.std(nonzero, ddof=0)) if nonzero_days > 0 else 0.0
    cv2 = float((nonzero_std / nonzero_mean) ** 2) if nonzero_days > 0 and nonzero_mean > 0 else 0.0
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


def build_profile_frame(panel: dict[str, object]) -> pd.DataFrame:
    sales = np.asarray(panel["sales"], dtype=float)
    metadata = pd.DataFrame(panel["metadata"]).copy().reset_index(drop=True)
    rows = []
    for idx, meta in metadata.iterrows():
        row = {
            "id": meta["id"],
            "cat_id": meta["cat_id"],
            "dept_id": meta["dept_id"],
            "store_id": meta["store_id"],
            "state_id": meta["state_id"],
        }
        row.update(profile_series(sales[idx]))
        rows.append(row)
    return pd.DataFrame(rows)


def _positive_corr(frame: pd.DataFrame) -> pd.DataFrame:
    corr = frame.corr(method="pearson").fillna(0.0)
    corr = corr.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def build_sales_similarity_components(sales_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    daily_corr = _positive_corr(sales_df)
    weekly_sales_df = sales_df.resample("W").sum()
    weekly_corr = _positive_corr(weekly_sales_df)
    log_corr = _positive_corr(np.log1p(sales_df))
    hybrid_sales = 0.5 * daily_corr + 0.3 * weekly_corr + 0.2 * log_corr
    np.fill_diagonal(hybrid_sales.values, 1.0)
    return {
        "daily_corr": daily_corr,
        "weekly_corr": weekly_corr,
        "log_corr": log_corr,
        "sales_similarity": hybrid_sales,
    }


def build_metadata_similarity(metadata: pd.DataFrame) -> pd.DataFrame:
    ids = metadata["id"].astype(str).tolist()
    values = np.zeros((len(ids), len(ids)), dtype=float)
    max_score = 4.0
    for i, left in metadata.iterrows():
        for j, right in metadata.iterrows():
            if i == j:
                values[i, j] = 1.0
                continue
            score = 0.0
            if left["cat_id"] == right["cat_id"]:
                score += 1.0
            if left["dept_id"] == right["dept_id"]:
                score += 1.5
            if left["store_id"] == right["store_id"]:
                score += 1.0
            if left["state_id"] == right["state_id"]:
                score += 0.5
            values[i, j] = score / max_score
    return pd.DataFrame(values, index=ids, columns=ids)


def build_profile_similarity(profile_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["mean_sales", "std_sales", "zero_rate", "nonzero_days", "max_sales", "ADI", "CV2"]
    feature_df = profile_df[feature_cols].copy()
    mins = feature_df.min(axis=0)
    spans = (feature_df.max(axis=0) - mins).replace(0.0, 1.0)
    scaled = (feature_df - mins) / spans
    values = scaled.to_numpy(dtype=float)
    dist = np.sqrt(((values[:, None, :] - values[None, :, :]) ** 2).sum(axis=2))
    max_dist = float(np.sqrt(len(feature_cols)))
    sim = 1.0 - (dist / max(max_dist, 1e-6))
    sim = np.clip(sim, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    ids = profile_df["id"].astype(str).tolist()
    return pd.DataFrame(sim, index=ids, columns=ids)


def combine_hybrid_similarity(
    sales_similarity: pd.DataFrame,
    metadata_similarity: pd.DataFrame,
    profile_similarity: pd.DataFrame,
    weights: tuple[float, float, float] = (0.40, 0.35, 0.25),
) -> pd.DataFrame:
    sales_w, metadata_w, profile_w = weights
    hybrid = (
        sales_w * sales_similarity
        + metadata_w * metadata_similarity
        + profile_w * profile_similarity
    )
    hybrid = hybrid.clip(lower=0.0, upper=1.0)
    np.fill_diagonal(hybrid.values, 1.0)
    return hybrid


def render_similarity_heatmap(matrix_df: pd.DataFrame, figure_path: Path, title: str) -> None:
    labels = matrix_df.columns.tolist()
    values = matrix_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(values, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] >= 0.6 else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Similarity")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def hybrid_threshold_summary(hybrid_df: pd.DataFrame, thresholds: list[float] | None = None) -> pd.DataFrame:
    thresholds = [0.5, 0.8, 0.9, 0.99] if thresholds is None else thresholds
    pair_df = compute_top_pairs(hybrid_df, value_col="similarity")
    total_pairs = int(len(pair_df))
    rows = []
    for threshold in thresholds:
        edge_count = int((pair_df["similarity"] >= float(threshold)).sum())
        graph = nx.Graph()
        graph.add_nodes_from(hybrid_df.columns.tolist())
        for row in pair_df.itertuples(index=False):
            if row.similarity >= float(threshold):
                graph.add_edge(row.product_a, row.product_b, weight=row.similarity)
        components = list(nx.connected_components(graph))
        largest_component = max((len(c) for c in components), default=0)
        rows.append(
            {
                "threshold": float(threshold),
                "edge_count": edge_count,
                "pair_coverage": float(edge_count / max(total_pairs, 1)),
                "connected_components": int(len(components)),
                "largest_component_size": int(largest_component),
            }
        )
    return pd.DataFrame(rows)


def build_neighbor_rank_table(hybrid_df: pd.DataFrame) -> pd.DataFrame:
    ids = hybrid_df.index.tolist()
    rows: list[dict[str, object]] = []
    for source in ids:
        scores = hybrid_df.loc[source].drop(index=source).sort_values(ascending=False)
        for rank, (target, similarity) in enumerate(scores.items(), start=1):
            rows.append(
                {
                    "source_product": source,
                    "target_product": target,
                    "hybrid_similarity": float(similarity),
                    "neighbor_rank": int(rank),
                }
            )
    return pd.DataFrame(rows)


def build_edge_table(
    hybrid_df: pd.DataFrame,
    top_k: int = 3,
    formal_threshold: float = 0.5,
) -> pd.DataFrame:
    neighbor_table = build_neighbor_rank_table(hybrid_df)
    top_k_table = neighbor_table.loc[neighbor_table["neighbor_rank"] <= int(top_k)].copy()
    reverse_lookup = {
        (row.source_product, row.target_product): row.neighbor_rank
        for row in top_k_table.itertuples(index=False)
    }

    pair_rows: list[dict[str, object]] = []
    ids = hybrid_df.index.tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            left = ids[i]
            right = ids[j]
            similarity = float(hybrid_df.loc[left, right])
            left_rank = reverse_lookup.get((left, right))
            right_rank = reverse_lookup.get((right, left))
            retained_topk = (left_rank is not None) or (right_rank is not None)
            pair_rows.append(
                {
                    "source_product": left,
                    "target_product": right,
                    "hybrid_similarity": similarity,
                    "edge_retained_threshold_0_50": "yes" if similarity >= formal_threshold else "no",
                    "edge_retained_top_k": "yes" if retained_topk else "no",
                    "source_neighbor_rank": left_rank,
                    "target_neighbor_rank": right_rank,
                    "best_neighbor_rank": min(
                        [rank for rank in [left_rank, right_rank] if rank is not None],
                        default=np.nan,
                    ),
                }
            )
    edge_df = pd.DataFrame(pair_rows).sort_values(
        ["hybrid_similarity", "best_neighbor_rank"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    return edge_df


def build_topk_graph(hybrid_df: pd.DataFrame, top_k: int = 3) -> nx.Graph:
    graph = nx.Graph()
    ids = hybrid_df.index.tolist()
    graph.add_nodes_from(ids)
    for source in ids:
        scores = hybrid_df.loc[source].drop(index=source).sort_values(ascending=False)
        for rank, (target, similarity) in enumerate(scores.head(int(top_k)).items(), start=1):
            if graph.has_edge(source, target):
                existing = graph[source][target]
                existing["weight"] = max(existing["weight"], float(similarity))
                existing["source_rank"] = min(existing.get("source_rank", rank), rank)
            else:
                graph.add_edge(
                    source,
                    target,
                    weight=float(similarity),
                    source_rank=int(rank),
                )
    return graph


def build_threshold_graph(hybrid_df: pd.DataFrame, threshold: float = 0.5) -> nx.Graph:
    graph = nx.Graph()
    ids = hybrid_df.index.tolist()
    graph.add_nodes_from(ids)
    for row in compute_top_pairs(hybrid_df, value_col="similarity").itertuples(index=False):
        if row.similarity >= float(threshold):
            graph.add_edge(row.product_a, row.product_b, weight=float(row.similarity))
    return graph


def _graph_style_maps(metadata: pd.DataFrame) -> tuple[dict[str, str], dict[str, tuple[float, float, float, float]], dict[str, float]]:
    ids = metadata["id"].astype(str).tolist()
    cat_ids = metadata["cat_id"].astype(str).tolist()
    mean_sales = metadata["mean_sales"].astype(float).to_numpy()
    unique_cats = sorted(set(cat_ids))
    palette = plt.cm.get_cmap("tab20", max(len(unique_cats), 1))
    cat_to_color = {cat: palette(i) for i, cat in enumerate(unique_cats)}
    max_mean = float(np.max(mean_sales)) if len(mean_sales) else 1.0
    node_sizes = {
        sid: 500 + 3500 * (metadata.loc[metadata["id"] == sid, "mean_sales"].iloc[0] / max(max_mean, 1e-6))
        for sid in ids
    }
    node_color_map = {
        sid: cat_to_color[metadata.loc[metadata["id"] == sid, "cat_id"].iloc[0]]
        for sid in ids
    }
    label_map = {sid: sid[:20] for sid in ids}
    return label_map, node_color_map, node_sizes


def render_labeled_graph(
    graph: nx.Graph,
    metadata: pd.DataFrame,
    figure_path: Path,
    title: str,
    layout_mode: str = "spring",
    spring_k: float | None = None,
    iterations: int = 200,
) -> None:
    label_map, node_color_map, node_sizes = _graph_style_maps(metadata)
    if layout_mode == "kamada_kawai":
        layout = nx.kamada_kawai_layout(graph, weight="weight")
    else:
        layout = nx.spring_layout(graph, seed=42, weight="weight", k=spring_k, iterations=iterations)
    node_colors = [node_color_map[node] for node in graph.nodes()]
    edge_widths = [1.0 + 7.0 * graph[u][v].get("weight", 0.0) for u, v in graph.edges()]
    edge_labels = {(u, v): f"{graph[u][v].get('weight', 0.0):.2f}" for u, v in graph.edges()}

    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_edges(graph, layout, width=edge_widths, edge_color="gray", alpha=0.65, ax=ax)
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_size=[node_sizes[node] for node in graph.nodes()],
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.0,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        labels={node: label_map[node] for node in graph.nodes()},
        font_size=8,
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        layout,
        edge_labels=edge_labels,
        font_size=7,
        rotate=False,
        bbox={"alpha": 0.7, "facecolor": "white", "edgecolor": "none", "pad": 0.2},
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def component_aware_layout(
    graph: nx.Graph,
    seed: int = 42,
    component_spacing_x: float = 5.0,
    component_spacing_y: float = 4.0,
) -> dict[str, np.ndarray]:
    components = [graph.subgraph(nodes).copy() for nodes in nx.connected_components(graph)]
    components = sorted(components, key=lambda g: (-g.number_of_nodes(), sorted(g.nodes())[0]))
    if not components:
        return {}

    n_components = len(components)
    n_cols = int(np.ceil(np.sqrt(n_components)))
    positions: dict[str, np.ndarray] = {}

    for idx, subgraph in enumerate(components):
        row = idx // n_cols
        col = idx % n_cols
        center = np.array([col * component_spacing_x, -row * component_spacing_y], dtype=float)

        if subgraph.number_of_nodes() == 1:
            local_pos = {next(iter(subgraph.nodes())): np.array([0.0, 0.0], dtype=float)}
        elif subgraph.number_of_edges() == 0:
            local_pos = nx.circular_layout(subgraph, scale=0.8)
        else:
            local_pos = nx.spring_layout(subgraph, seed=seed, weight="weight", k=1.0, iterations=400)

        local_values = np.array(list(local_pos.values()), dtype=float)
        if len(local_values) > 0:
            local_values = local_values - local_values.mean(axis=0, keepdims=True)
        for node, point in zip(local_pos.keys(), local_values):
            positions[node] = center + point
    return positions


def render_component_aware_graph(
    graph: nx.Graph,
    metadata: pd.DataFrame,
    figure_path: Path,
    title: str,
) -> None:
    label_map, node_color_map, node_sizes = _graph_style_maps(metadata)
    layout = component_aware_layout(graph)
    node_colors = [node_color_map[node] for node in graph.nodes()]
    adjusted_sizes = [max(420.0, node_sizes[node] * 0.72) for node in graph.nodes()]
    edge_widths = [2.0 + 9.0 * graph[u][v].get("weight", 0.0) for u, v in graph.edges()]
    edge_labels = {(u, v): f"{graph[u][v].get('weight', 0.0):.2f}" for u, v in graph.edges()}

    fig, ax = plt.subplots(figsize=(18, 13))
    nx.draw_networkx_edges(
        graph,
        layout,
        width=edge_widths,
        edge_color="#3a3a3a",
        alpha=0.9,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_size=adjusted_sizes,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.2,
        ax=ax,
    )
    label_pos = {node: point + np.array([0.0, 0.18]) for node, point in layout.items()}
    nx.draw_networkx_labels(
        graph,
        label_pos,
        labels={node: label_map[node] for node in graph.nodes()},
        font_size=8,
        font_weight="bold",
        ax=ax,
    )
    nx.draw_networkx_edge_labels(
        graph,
        layout,
        edge_labels=edge_labels,
        font_size=8,
        rotate=False,
        label_pos=0.55,
        bbox={"alpha": 0.92, "facecolor": "#fff9e8", "edgecolor": "#666666", "boxstyle": "round,pad=0.2"},
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def render_hybrid_graph(
    hybrid_df: pd.DataFrame,
    metadata: pd.DataFrame,
    figure_path: Path,
    threshold: float = 0.5,
) -> None:
    graph = build_threshold_graph(hybrid_df, threshold=threshold)
    for _, meta in metadata.iterrows():
        graph.nodes[meta["id"]]["cat_id"] = meta["cat_id"]
    render_labeled_graph(graph, metadata, figure_path, f"Hybrid Product Graph (threshold >= {threshold:.2f})")


def render_heatmap(corr_df: pd.DataFrame, figure_path: Path) -> None:
    labels = corr_df.columns.tolist()
    values = corr_df.to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("M5 16-Product Pearson Correlation Matrix")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if abs(values[i, j]) >= 0.5 else "black"
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_interpretation(
    metadata: pd.DataFrame,
    corr_df: pd.DataFrame,
    top_pairs_df: pd.DataFrame,
    connectivity_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
) -> list[str]:
    top_positive = top_pairs_df.head(5)
    category_scores = (
        connectivity_df.loc[connectivity_df["pair_type"] == "within_category", ["category", "mean_correlation"]]
        .sort_values("mean_correlation", ascending=False)
        .reset_index(drop=True)
    )
    node_strength = corr_df.abs().sum(axis=1) - 1.0
    node_strength = node_strength.sort_values(ascending=False)

    lines = [
        "The same 16-product subset from notebook29 was recovered through load_m5_panel_subset with the original CA / seed=42 / max_days=365 configuration.",
        "The strongest product-to-product relationships are concentrated in a small set of positive pairs, which supports threshold-based graph construction from the correlation matrix.",
        "Top correlated pairs: "
        + "; ".join(
            f"{row.product_a} vs {row.product_b} = {row.correlation:.3f}"
            for row in top_positive.itertuples(index=False)
        ),
        "Within-category average connectivity ranks as "
        + ", ".join(
            f"{row.category}: {row.mean_correlation:.3f}"
            for row in category_scores.itertuples(index=False)
        )
        + ".",
        "Hub-like products by absolute correlation strength are "
        + ", ".join(f"{idx} ({value:.3f})" for idx, value in node_strength.head(5).items())
        + ".",
        "Correlation-only threshold counts are "
        + ", ".join(
            f"{row.threshold:.2f}: {int(row.edge_count)} edges"
            for row in threshold_df.itertuples(index=False)
        )
        + ", which means a pure correlation graph at 0.5 or above would be empty for this subset and window.",
    ]
    return lines


def build_hybrid_interpretation(
    hybrid_top_pairs_df: pd.DataFrame,
    hybrid_connectivity_df: pd.DataFrame,
    hybrid_threshold_df: pd.DataFrame,
    weights: tuple[float, float, float],
) -> list[str]:
    top_pairs = hybrid_top_pairs_df.head(5)
    within_scores = (
        hybrid_connectivity_df.loc[
            hybrid_connectivity_df["pair_type"] == "within_category",
            ["category", "mean_similarity"],
        ]
        .sort_values("mean_similarity", ascending=False)
        .reset_index(drop=True)
    )
    return [
        "Hybrid similarity was defined as 0.40 * sales_similarity + 0.35 * metadata_similarity + 0.25 * profile_similarity.",
        "The sales block combines positive daily Pearson, weekly Pearson, and log1p Pearson to stabilize sparse daily series.",
        "The metadata block reuses the same category / department / store / state scoring idea from notebook29.",
        "The demand-profile block connects products with similar mean, spread, sparsity, and intermittency characteristics even when raw day-level correlation is weak.",
        "Top hybrid pairs: "
        + "; ".join(
            f"{row.product_a} vs {row.product_b} = {row.similarity:.3f}"
            for row in top_pairs.itertuples(index=False)
        ),
        "Within-category hybrid connectivity ranks as "
        + ", ".join(
            f"{row.category}: {row.mean_similarity:.3f}"
            for row in within_scores.itertuples(index=False)
        )
        + ".",
        "Hybrid threshold counts are "
        + ", ".join(
            f"{row.threshold:.2f}: {int(row.edge_count)} edges"
            for row in hybrid_threshold_df.itertuples(index=False)
        )
        + ", so the hybrid graph remains meaningful at moderate thresholds even though the raw Pearson graph did not.",
    ]


def build_graph_interpretation(
    edge_table_df: pd.DataFrame,
    hybrid_threshold_df: pd.DataFrame,
    top_k: int,
    exploratory_threshold: float,
) -> list[str]:
    top_edges = edge_table_df.head(8)
    hub_scores = (
        pd.concat(
            [
                edge_table_df[["source_product", "hybrid_similarity"]].rename(columns={"source_product": "product"}),
                edge_table_df[["target_product", "hybrid_similarity"]].rename(columns={"target_product": "product"}),
            ],
            ignore_index=True,
        )
        .groupby("product", as_index=False)["hybrid_similarity"]
        .sum()
        .sort_values("hybrid_similarity", ascending=False)
        .reset_index(drop=True)
    )
    threshold_050 = hybrid_threshold_df.loc[hybrid_threshold_df["threshold"] == 0.5].iloc[0]
    return [
        f"The readable weighted graph uses a top-{top_k} neighbor rule per node, which preserves each product's strongest relationships without drawing the full dense 16-node graph.",
        "The formal threshold graph still uses similarity >= 0.50, so it stays sparse by design and mainly highlights the strongest retained pairs.",
        "Top weighted edges are "
        + "; ".join(
            f"{row.source_product} - {row.target_product} = {row.hybrid_similarity:.3f}"
            for row in top_edges.itertuples(index=False)
        ),
        "Hub-like products by total retained edge strength are "
        + ", ".join(
            f"{row.product} ({row.hybrid_similarity:.3f})"
            for row in hub_scores.head(5).itertuples(index=False)
        )
        + ".",
        f"At threshold 0.50 the graph keeps {int(threshold_050.edge_count)} edges and the largest connected component has size {int(threshold_050.largest_component_size)}, which explains why the formal graph remains visually sparse.",
        f"The exploratory threshold {exploratory_threshold:.2f} is included only for interpretation, to show medium-strength category-consistent links that are hidden by the stricter formal threshold.",
    ]


def build_final_graph_interpretation(
    edge_table_df: pd.DataFrame,
    top_k: int,
) -> list[str]:
    retained_df = edge_table_df.loc[edge_table_df["edge_retained_top_k"] == "yes"].copy()
    hub_scores = (
        pd.concat(
            [
                retained_df[["source_product", "hybrid_similarity"]].rename(columns={"source_product": "product"}),
                retained_df[["target_product", "hybrid_similarity"]].rename(columns={"target_product": "product"}),
            ],
            ignore_index=True,
        )
        .groupby("product", as_index=False)["hybrid_similarity"]
        .sum()
        .sort_values("hybrid_similarity", ascending=False)
        .reset_index(drop=True)
    )
    return [
        f"This final graph uses a top-{top_k} neighbors per product rule on the hybrid similarity matrix.",
        "It was chosen because threshold-only views were too sparse to represent all products clearly in one figure.",
        "The graph retains both within-category and cross-product links while keeping every product visible in a readable undirected structure.",
        "Most connected products by retained edge strength are "
        + ", ".join(
            f"{row.product} ({row.hybrid_similarity:.3f})"
            for row in hub_scores.head(5).itertuples(index=False)
        )
        + ".",
    ]


def main() -> dict[str, object]:
    config = CONFIG
    reports_dir = REPO_ROOT / "reports" / "gnn_benchmarks" / config.reports_subdir
    reports_dir.mkdir(parents=True, exist_ok=True)

    panel = load_gnn_subset(config)
    metadata = pd.DataFrame(panel["metadata"]).copy().reset_index(drop=True)
    sales_df = build_sales_frame(panel)
    corr_df = sales_df.corr(method=config.corr_method).fillna(0.0)
    np.fill_diagonal(corr_df.values, 1.0)

    top_pairs_df = compute_top_pairs(corr_df)
    connectivity_df = summarize_connectivity(metadata, corr_df)
    threshold_df = threshold_edge_summary(corr_df)

    profile_df = build_profile_frame(panel)
    sales_components = build_sales_similarity_components(sales_df)
    metadata_similarity_df = build_metadata_similarity(metadata)
    profile_similarity_df = build_profile_similarity(profile_df)
    hybrid_weights = (0.40, 0.35, 0.25)
    hybrid_df = combine_hybrid_similarity(
        sales_similarity=sales_components["sales_similarity"],
        metadata_similarity=metadata_similarity_df,
        profile_similarity=profile_similarity_df,
        weights=hybrid_weights,
    )
    hybrid_top_pairs_df = compute_top_pairs(hybrid_df, value_col="similarity")
    hybrid_connectivity_df = summarize_connectivity(metadata, hybrid_df, value_col="similarity")
    hybrid_threshold_df = hybrid_threshold_summary(hybrid_df)
    graph_top_k = 3
    exploratory_threshold = 0.35
    edge_table_df = build_edge_table(hybrid_df, top_k=graph_top_k, formal_threshold=0.5)
    topk_graph = build_topk_graph(hybrid_df, top_k=graph_top_k)
    exploratory_graph = build_threshold_graph(hybrid_df, threshold=exploratory_threshold)
    final_graph_top_k = 2
    final_edge_table_df = build_edge_table(hybrid_df, top_k=final_graph_top_k, formal_threshold=0.5)
    final_edge_table_df = final_edge_table_df.loc[final_edge_table_df["edge_retained_top_k"] == "yes"].reset_index(drop=True)
    final_top2_graph = build_topk_graph(hybrid_df, top_k=final_graph_top_k)

    corr_csv = reports_dir / "correlation_matrix.csv"
    sales_csv = reports_dir / "sales_panel_16_products.csv"
    top_pairs_csv = reports_dir / "top_correlated_pairs.csv"
    connectivity_csv = reports_dir / "category_connectivity_summary.csv"
    threshold_csv = reports_dir / "threshold_edge_summary.csv"
    figure_path = reports_dir / "correlation_heatmap.png"
    summary_path = reports_dir / "interpretation_summary.txt"
    profile_csv = reports_dir / "profile_features_16_products.csv"
    daily_component_csv = reports_dir / "daily_positive_correlation.csv"
    weekly_component_csv = reports_dir / "weekly_positive_correlation.csv"
    log_component_csv = reports_dir / "log1p_positive_correlation.csv"
    sales_similarity_csv = reports_dir / "sales_similarity_component.csv"
    metadata_similarity_csv = reports_dir / "metadata_similarity_component.csv"
    profile_similarity_csv = reports_dir / "profile_similarity_component.csv"
    hybrid_csv = reports_dir / "hybrid_similarity_matrix.csv"
    hybrid_top_pairs_csv = reports_dir / "hybrid_top_pairs.csv"
    hybrid_connectivity_csv = reports_dir / "hybrid_category_connectivity_summary.csv"
    hybrid_threshold_csv = reports_dir / "hybrid_threshold_edge_summary.csv"
    hybrid_heatmap_path = reports_dir / "hybrid_similarity_heatmap.png"
    hybrid_graph_path = reports_dir / "hybrid_graph_threshold_0_50.png"
    hybrid_summary_path = reports_dir / "hybrid_interpretation_summary.txt"
    graph_edge_table_csv = reports_dir / "hybrid_graph_edge_table.csv"
    weighted_topk_graph_path = reports_dir / "hybrid_weighted_topk_graph_with_labels.png"
    threshold_labeled_graph_path = reports_dir / "hybrid_threshold_0_50_graph_with_labels.png"
    exploratory_graph_path = reports_dir / "hybrid_exploratory_threshold_0_35_graph_with_labels.png"
    graph_summary_path = reports_dir / "hybrid_graph_interpretation_summary.txt"
    improved_threshold_graph_path = reports_dir / "hybrid_threshold_0_50_graph_compact_with_labels.png"
    exploratory_045_graph_path = reports_dir / "hybrid_exploratory_threshold_0_45_graph_with_labels.png"
    viz_summary_path = reports_dir / "hybrid_graph_visualization_notes.txt"
    clear_threshold_graph_path = reports_dir / "hybrid_threshold_0_50_graph_component_layout.png"
    clear_exploratory_045_graph_path = reports_dir / "hybrid_threshold_0_45_graph_component_layout.png"
    clear_viz_summary_path = reports_dir / "hybrid_graph_visualization_fix_notes.txt"
    final_top2_graph_path = reports_dir / "final_hybrid_product_relationship_graph_top2_neighbors.png"
    final_top2_edge_table_csv = reports_dir / "final_hybrid_top2_edge_table.csv"
    final_top2_summary_path = reports_dir / "final_hybrid_top2_graph_notes.txt"

    corr_df.to_csv(corr_csv, index=True)
    sales_df.to_csv(sales_csv, index=True)
    top_pairs_df.to_csv(top_pairs_csv, index=False)
    connectivity_df.to_csv(connectivity_csv, index=False)
    threshold_df.to_csv(threshold_csv, index=False)
    profile_df.to_csv(profile_csv, index=False)
    sales_components["daily_corr"].to_csv(daily_component_csv, index=True)
    sales_components["weekly_corr"].to_csv(weekly_component_csv, index=True)
    sales_components["log_corr"].to_csv(log_component_csv, index=True)
    sales_components["sales_similarity"].to_csv(sales_similarity_csv, index=True)
    metadata_similarity_df.to_csv(metadata_similarity_csv, index=True)
    profile_similarity_df.to_csv(profile_similarity_csv, index=True)
    hybrid_df.to_csv(hybrid_csv, index=True)
    hybrid_top_pairs_df.to_csv(hybrid_top_pairs_csv, index=False)
    hybrid_connectivity_df.to_csv(hybrid_connectivity_csv, index=False)
    hybrid_threshold_df.to_csv(hybrid_threshold_csv, index=False)
    edge_table_df.to_csv(graph_edge_table_csv, index=False)
    render_heatmap(corr_df, figure_path)
    render_similarity_heatmap(hybrid_df, hybrid_heatmap_path, "M5 16-Product Hybrid Similarity Matrix")
    render_hybrid_graph(hybrid_df, metadata, hybrid_graph_path, threshold=0.5)
    render_labeled_graph(
        topk_graph,
        metadata,
        weighted_topk_graph_path,
        f"Weighted Hybrid Product Graph (top-{graph_top_k} neighbors)",
    )
    render_labeled_graph(
        build_threshold_graph(hybrid_df, threshold=0.5),
        metadata,
        threshold_labeled_graph_path,
        "Hybrid Product Graph with Edge Labels (threshold >= 0.50)",
    )
    render_labeled_graph(
        exploratory_graph,
        metadata,
        exploratory_graph_path,
        f"Exploratory Hybrid Product Graph with Edge Labels (threshold >= {exploratory_threshold:.2f})",
    )
    render_labeled_graph(
        build_threshold_graph(hybrid_df, threshold=0.5),
        metadata,
        improved_threshold_graph_path,
        "Improved Hybrid Product Graph with Edge Labels (threshold >= 0.50)",
        layout_mode="spring",
        spring_k=0.48,
        iterations=400,
    )
    render_labeled_graph(
        build_threshold_graph(hybrid_df, threshold=0.45),
        metadata,
        exploratory_045_graph_path,
        "Exploratory Hybrid Product Graph with Edge Labels (threshold >= 0.45)",
        layout_mode="spring",
        spring_k=0.42,
        iterations=400,
    )
    render_component_aware_graph(
        build_threshold_graph(hybrid_df, threshold=0.5),
        metadata,
        clear_threshold_graph_path,
        "Clear Hybrid Product Graph with Edge Labels (threshold >= 0.50)",
    )
    render_component_aware_graph(
        build_threshold_graph(hybrid_df, threshold=0.45),
        metadata,
        clear_exploratory_045_graph_path,
        "Clear Exploratory Hybrid Product Graph with Edge Labels (threshold >= 0.45)",
    )
    render_component_aware_graph(
        final_top2_graph,
        metadata,
        final_top2_graph_path,
        "Final Hybrid Product Relationship Graph (Top-2 Neighbors)",
    )

    interpretation_lines = build_interpretation(metadata, corr_df, top_pairs_df, connectivity_df, threshold_df)
    hybrid_interpretation_lines = build_hybrid_interpretation(
        hybrid_top_pairs_df=hybrid_top_pairs_df,
        hybrid_connectivity_df=hybrid_connectivity_df,
        hybrid_threshold_df=hybrid_threshold_df,
        weights=hybrid_weights,
    )
    graph_interpretation_lines = build_graph_interpretation(
        edge_table_df=edge_table_df,
        hybrid_threshold_df=hybrid_threshold_df,
        top_k=graph_top_k,
        exploratory_threshold=exploratory_threshold,
    )
    visualization_lines = [
        "The improved formal graph keeps the exact same threshold 0.50 edge set as before; only the layout was compacted to reduce empty space and make labels easier to read.",
        "The exploratory graph uses threshold 0.45, not to replace the formal result, but to surface medium-strength relationships that are visually hidden at 0.50.",
        "The 0.50 graph remains sparse because only a small number of hybrid similarities clear that stricter cutoff.",
        "The 0.45 view reveals a denser FOODS-centered backbone and makes within-category neighborhoods easier to inspect, which is closer in spirit to the relationship view from notebook29.",
    ]
    clear_visualization_lines = [
        "This visualization-fix section keeps the same threshold 0.50 graph result and only changes the drawing strategy.",
        "Connected components are laid out separately and then placed apart on a grid, which keeps edges from being hidden behind large nodes and reduces wasted space.",
        "Edges are darker and thicker, and edge labels are drawn in boxed annotations so weights such as 0.55 and 0.60 remain readable.",
        "The 0.45 figure is still exploratory only; it reveals additional medium-strength relationships while preserving the same hybrid similarity matrix.",
    ]
    final_graph_lines = build_final_graph_interpretation(
        edge_table_df=final_edge_table_df,
        top_k=final_graph_top_k,
    )
    summary_path.write_text("\n".join(interpretation_lines), encoding="utf-8")
    hybrid_summary_path.write_text("\n".join(hybrid_interpretation_lines), encoding="utf-8")
    graph_summary_path.write_text("\n".join(graph_interpretation_lines), encoding="utf-8")
    viz_summary_path.write_text("\n".join(visualization_lines), encoding="utf-8")
    clear_viz_summary_path.write_text("\n".join(clear_visualization_lines), encoding="utf-8")
    final_top2_edge_table_df = final_edge_table_df.copy()
    final_top2_edge_table_df.to_csv(final_top2_edge_table_csv, index=False)
    final_top2_summary_path.write_text("\n".join(final_graph_lines), encoding="utf-8")

    return {
        "config": config,
        "panel": panel,
        "metadata": metadata,
        "sales_df": sales_df,
        "corr_df": corr_df,
        "top_pairs_df": top_pairs_df,
        "connectivity_df": connectivity_df,
        "threshold_df": threshold_df,
        "profile_df": profile_df,
        "daily_corr_df": sales_components["daily_corr"],
        "weekly_corr_df": sales_components["weekly_corr"],
        "log_corr_df": sales_components["log_corr"],
        "sales_similarity_df": sales_components["sales_similarity"],
        "metadata_similarity_df": metadata_similarity_df,
        "profile_similarity_df": profile_similarity_df,
        "hybrid_df": hybrid_df,
        "hybrid_top_pairs_df": hybrid_top_pairs_df,
        "hybrid_connectivity_df": hybrid_connectivity_df,
        "hybrid_threshold_df": hybrid_threshold_df,
        "hybrid_weights": hybrid_weights,
        "edge_table_df": edge_table_df,
        "graph_top_k": graph_top_k,
        "exploratory_threshold": exploratory_threshold,
        "reports_dir": reports_dir,
        "corr_csv": corr_csv,
        "sales_csv": sales_csv,
        "top_pairs_csv": top_pairs_csv,
        "connectivity_csv": connectivity_csv,
        "threshold_csv": threshold_csv,
        "figure_path": figure_path,
        "summary_path": summary_path,
        "interpretation_lines": interpretation_lines,
        "profile_csv": profile_csv,
        "daily_component_csv": daily_component_csv,
        "weekly_component_csv": weekly_component_csv,
        "log_component_csv": log_component_csv,
        "sales_similarity_csv": sales_similarity_csv,
        "metadata_similarity_csv": metadata_similarity_csv,
        "profile_similarity_csv": profile_similarity_csv,
        "hybrid_csv": hybrid_csv,
        "hybrid_top_pairs_csv": hybrid_top_pairs_csv,
        "hybrid_connectivity_csv": hybrid_connectivity_csv,
        "hybrid_threshold_csv": hybrid_threshold_csv,
        "hybrid_heatmap_path": hybrid_heatmap_path,
        "hybrid_graph_path": hybrid_graph_path,
        "hybrid_summary_path": hybrid_summary_path,
        "hybrid_interpretation_lines": hybrid_interpretation_lines,
        "graph_edge_table_csv": graph_edge_table_csv,
        "weighted_topk_graph_path": weighted_topk_graph_path,
        "threshold_labeled_graph_path": threshold_labeled_graph_path,
        "exploratory_graph_path": exploratory_graph_path,
        "graph_summary_path": graph_summary_path,
        "graph_interpretation_lines": graph_interpretation_lines,
        "improved_threshold_graph_path": improved_threshold_graph_path,
        "exploratory_045_graph_path": exploratory_045_graph_path,
        "viz_summary_path": viz_summary_path,
        "visualization_lines": visualization_lines,
        "clear_threshold_graph_path": clear_threshold_graph_path,
        "clear_exploratory_045_graph_path": clear_exploratory_045_graph_path,
        "clear_viz_summary_path": clear_viz_summary_path,
        "clear_visualization_lines": clear_visualization_lines,
        "final_graph_top_k": final_graph_top_k,
        "final_top2_graph_path": final_top2_graph_path,
        "final_top2_edge_table_csv": final_top2_edge_table_csv,
        "final_top2_summary_path": final_top2_summary_path,
        "final_top2_edge_table_df": final_top2_edge_table_df,
        "final_graph_lines": final_graph_lines,
    }


if __name__ == "__main__":
    outputs = main()
    print("Saved correlation matrix to:", outputs["corr_csv"])
    print("Saved sales panel to:", outputs["sales_csv"])
    print("Saved heatmap to:", outputs["figure_path"])
