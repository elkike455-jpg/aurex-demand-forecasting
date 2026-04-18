from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.deep_challenger_models import _as_float_array, _choose_device


def _node_scale_matrix(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    scales = []
    for row in y:
        nz = np.abs(row[row != 0])
        scale = float(np.mean(nz)) if nz.size else 1.0
        scales.append(max(scale, 1.0))
    return np.asarray(scales, dtype=np.float32)


def build_graph_windows(panel: np.ndarray, context_length: int) -> Tuple[np.ndarray, np.ndarray]:
    panel = np.asarray(panel, dtype=np.float32)
    n_nodes, total_steps = panel.shape
    if total_steps <= context_length:
        raise ValueError(
            f"Panel length {total_steps} must be greater than context_length {context_length}."
        )

    samples = total_steps - context_length
    X = np.zeros((samples, n_nodes, context_length, 1), dtype=np.float32)
    y = np.zeros((samples, n_nodes), dtype=np.float32)
    for idx in range(samples):
        start = idx
        end = idx + context_length
        X[idx, :, :, 0] = panel[:, start:end]
        y[idx] = panel[:, end]
    return X, y


def build_m5_product_graph(
    metadata: "pd.DataFrame",
    train_sales: np.ndarray,
    top_k: int = 5,
    corr_weight: float = 0.35,
) -> np.ndarray:
    """
    Build a simple weighted graph using M5 product metadata plus demand correlation.
    """

    n_nodes = len(metadata)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 1.0)

    sales = np.asarray(train_sales, dtype=float)
    if sales.ndim != 2 or sales.shape[0] != n_nodes:
        raise ValueError("train_sales must have shape (n_nodes, T) aligned with metadata.")

    if sales.shape[1] >= 3:
        corr = np.corrcoef(sales)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        corr = np.zeros((n_nodes, n_nodes), dtype=float)

    for i in range(n_nodes):
        scores = np.zeros(n_nodes, dtype=np.float32)
        for j in range(n_nodes):
            if i == j:
                continue
            score = 0.0
            if metadata.iloc[i]["cat_id"] == metadata.iloc[j]["cat_id"]:
                score += 1.0
            if metadata.iloc[i]["dept_id"] == metadata.iloc[j]["dept_id"]:
                score += 1.5
            if metadata.iloc[i]["store_id"] == metadata.iloc[j]["store_id"]:
                score += 1.0
            if metadata.iloc[i]["state_id"] == metadata.iloc[j]["state_id"]:
                score += 0.5
            corr_ij = max(float(corr[i, j]), 0.0)
            score += float(corr_weight) * corr_ij
            scores[j] = float(score)

        if top_k > 0:
            keep = np.argsort(scores)[-int(top_k) :]
            for j in keep:
                if scores[j] > 0:
                    adj[i, j] = scores[j]

    adj = np.maximum(adj, adj.T)
    degree = adj.sum(axis=1, keepdims=True)
    degree[degree <= 0] = 1.0
    return (adj / degree).astype(np.float32)


class _MeanGraphLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, node_repr: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        if node_repr.ndim != 3:
            raise ValueError(f"node_repr must have shape (batch, n_nodes, hidden), got {tuple(node_repr.shape)}")
        neigh = torch.matmul(adj.unsqueeze(0), node_repr)
        return self.activation(self.linear(neigh))


class _PointHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, local_repr: torch.Tensor, graph_repr: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([local_repr, graph_repr], dim=-1)
        return self.net(fused).squeeze(-1)


class _HurdleHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.occurrence = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
        self.size = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
        )

    def forward(self, local_repr: torch.Tensor, graph_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fused = torch.cat([local_repr, graph_repr], dim=-1)
        logits = self.occurrence(fused).squeeze(-1)
        size = self.size(fused).squeeze(-1)
        return logits, size


class _GraphDemandNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        hurdle_mode: bool,
        num_graph_layers: int,
    ):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.temporal = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=gru_dropout,
            batch_first=True,
        )
        self.graph_layers = nn.ModuleList(
            [_MeanGraphLayer(hidden_size=hidden_size) for _ in range(max(1, int(num_graph_layers)))]
        )
        self.hurdle_mode = hurdle_mode
        if hurdle_mode:
            self.head = _HurdleHead(hidden_size=hidden_size)
        else:
            self.head = _PointHead(hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, adj: torch.Tensor):
        batch_size, n_nodes, context_length, n_features = x.shape
        if n_features != 1:
            raise ValueError(f"Expected one temporal feature, got {n_features}")

        flat_x = x.view(batch_size * n_nodes, context_length, n_features)
        encoded, _ = self.temporal(flat_x)
        local_repr = encoded[:, -1, :].view(batch_size, n_nodes, -1)
        graph_repr = local_repr
        for graph_layer in self.graph_layers:
            graph_repr = graph_layer(graph_repr, adj)
        return self.head(local_repr, graph_repr)


@dataclass
class GraphTrainingConfig:
    context_length: int = 28
    hidden_size: int = 32
    num_layers: int = 1
    dropout: float = 0.0
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-5
    device: str | None = None
    seed: int = 42
    hurdle_mode: bool = False
    num_graph_layers: int = 1
    loss_name: str = "mse"


class GraphDemandForecastModel:
    """
    Lightweight graph forecaster:
      1. GRU temporal encoder per product
      2. one mean-aggregation graph layer across products
      3. point forecast head or optional hurdle-style head
    """

    def __init__(self, **kwargs):
        # Backward-compatible alias for notebook experiments that request
        # the hurdle decomposition explicitly as use_hurdle_head=True.
        if "use_hurdle_head" in kwargs and "hurdle_mode" not in kwargs:
            kwargs["hurdle_mode"] = kwargs.pop("use_hurdle_head")
        self.config = GraphTrainingConfig(**kwargs)
        self.device = _choose_device(self.config.device)
        self.model = _GraphDemandNet(
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            hurdle_mode=self.config.hurdle_mode,
            num_graph_layers=self.config.num_graph_layers,
        ).to(self.device)
        self.node_scale_: np.ndarray | None = None
        self.train_resid_std_: np.ndarray | None = None
        self.training_history_: Dict[str, float] = {}
        self.adj_: np.ndarray | None = None

    def fit(self, train_panel: np.ndarray, adj: np.ndarray) -> "GraphDemandForecastModel":
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        panel = np.asarray(train_panel, dtype=float)
        if panel.ndim != 2:
            raise ValueError("train_panel must have shape (n_nodes, T)")

        self.node_scale_ = _node_scale_matrix(panel)
        panel_scaled = (panel / self.node_scale_[:, None]).astype(np.float32)
        X, target = build_graph_windows(panel_scaled, self.config.context_length)

        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(target))
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        adj_tensor = torch.from_numpy(np.asarray(adj, dtype=np.float32)).to(self.device)
        self.adj_ = np.asarray(adj, dtype=np.float32)

        final_loss = None
        self.model.train()
        for _ in range(self.config.epochs):
            epoch_losses = []
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                pred = self.model(batch_x, adj_tensor)
                if self.config.hurdle_mode:
                    logits, size = pred
                    occ_target = (batch_y > 0).float()
                    occ_loss = nn.functional.binary_cross_entropy_with_logits(logits, occ_target)
                    pos_mask = occ_target > 0
                    if pos_mask.any():
                        size_loss = nn.functional.mse_loss(size[pos_mask], batch_y[pos_mask])
                    else:
                        size_loss = torch.zeros((), device=self.device)
                    loss = occ_loss + 0.5 * size_loss
                else:
                    if self.config.loss_name == "huber":
                        loss = nn.functional.huber_loss(pred, batch_y, delta=1.0)
                    else:
                        loss = nn.functional.mse_loss(pred, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            final_loss = float(np.mean(epoch_losses)) if epoch_losses else None

        fitted_scaled = self._predict_in_sample_scaled(X)
        fitted = fitted_scaled * self.node_scale_[None, :]
        resid = panel[:, self.config.context_length :] - fitted.T
        self.train_resid_std_ = np.std(resid, axis=1).astype(np.float32)
        self.training_history_ = {"final_loss": final_loss if final_loss is not None else float("nan")}
        return self

    def forecast_components(self, h: int, history_panel: np.ndarray, adj: np.ndarray | None = None):
        if self.node_scale_ is None:
            raise RuntimeError("Model must be fitted before forecast.")
        adj_arr = self.adj_ if adj is None else np.asarray(adj, dtype=np.float32)
        adj_tensor = torch.from_numpy(adj_arr).to(self.device)

        history_scaled = (_as_float_array(history_panel) / self.node_scale_[:, None]).astype(np.float32)
        work = history_scaled.copy()
        preds_scaled = np.zeros((history_scaled.shape[0], int(h)), dtype=np.float32)
        occ_probs = np.full((history_scaled.shape[0], int(h)), np.nan, dtype=np.float32)
        magnitudes = np.full((history_scaled.shape[0], int(h)), np.nan, dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for step in range(int(h)):
                context = work[:, -self.config.context_length :]
                if context.shape[1] < self.config.context_length:
                    pad = np.zeros((context.shape[0], self.config.context_length - context.shape[1]), dtype=np.float32)
                    context = np.concatenate([pad, context], axis=1)
                x = torch.from_numpy(context[:, :, None]).unsqueeze(0).to(self.device)
                pred = self.model(x, adj_tensor)
                if self.config.hurdle_mode:
                    logits, size = pred
                    prob = torch.sigmoid(logits).squeeze(0)
                    mag = size.squeeze(0)
                    next_scaled = prob * mag
                    occ_probs[:, step] = prob.detach().cpu().numpy().astype(np.float32)
                    magnitudes[:, step] = mag.detach().cpu().numpy().astype(np.float32)
                else:
                    next_scaled = pred.squeeze(0)
                next_np = next_scaled.detach().cpu().numpy().astype(np.float32)
                preds_scaled[:, step] = next_np
                work = np.concatenate([work, next_np[:, None]], axis=1)

        y_pred = np.maximum(preds_scaled * self.node_scale_[:, None], 0.0)
        magnitude = np.maximum(magnitudes * self.node_scale_[:, None], 0.0)
        return {
            "prediction": y_pred,
            "occurrence_probability": occ_probs,
            "magnitude": magnitude,
        }

    def _predict_in_sample_scaled(self, X: np.ndarray) -> np.ndarray:
        if self.adj_ is None:
            raise RuntimeError("Model must be fitted before prediction.")
        adj_tensor = torch.from_numpy(self.adj_).to(self.device)
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.from_numpy(X).to(self.device)
            pred = self.model(tensor_x, adj_tensor)
            if self.config.hurdle_mode:
                logits, size = pred
                pred = torch.sigmoid(logits) * size
            pred_np = pred.detach().cpu().numpy()
        return pred_np.astype(np.float32)

    def forecast(self, h: int, history_panel: np.ndarray, adj: np.ndarray | None = None):
        if self.node_scale_ is None:
            raise RuntimeError("Model must be fitted before forecast.")
        adj_arr = self.adj_ if adj is None else np.asarray(adj, dtype=np.float32)
        adj_tensor = torch.from_numpy(adj_arr).to(self.device)

        history_scaled = (_as_float_array(history_panel) / self.node_scale_[:, None]).astype(np.float32)
        work = history_scaled.copy()
        preds_scaled = np.zeros((history_scaled.shape[0], int(h)), dtype=np.float32)

        self.model.eval()
        with torch.no_grad():
            for step in range(int(h)):
                context = work[:, -self.config.context_length :]
                if context.shape[1] < self.config.context_length:
                    pad = np.zeros((context.shape[0], self.config.context_length - context.shape[1]), dtype=np.float32)
                    context = np.concatenate([pad, context], axis=1)
                x = torch.from_numpy(context[:, :, None]).unsqueeze(0).to(self.device)
                pred = self.model(x, adj_tensor)
                if self.config.hurdle_mode:
                    logits, size = pred
                    next_scaled = (torch.sigmoid(logits) * size).squeeze(0)
                else:
                    next_scaled = pred.squeeze(0)
                next_np = next_scaled.detach().cpu().numpy().astype(np.float32)
                preds_scaled[:, step] = next_np
                work = np.concatenate([work, next_np[:, None]], axis=1)

        y_pred = np.maximum(preds_scaled * self.node_scale_[:, None], 0.0)
        resid_std = self.train_resid_std_ if self.train_resid_std_ is not None else np.zeros(y_pred.shape[0])
        spread = 1.959963984540054 * np.maximum(resid_std[:, None], 1e-6)
        conf_low = np.maximum(y_pred - spread, 0.0)
        conf_up = y_pred + spread
        return y_pred, conf_low, conf_up
