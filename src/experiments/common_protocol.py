from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class BenchmarkProtocol:
    """
    Official protocol for the M5 GNN benchmark subset and the three benchmark products.

    Scope:
      - same 16 M5 products used by notebook29
      - same three benchmark products selected from that subset
      - fair model-to-model comparison under one fixed chronological split
    """

    dataset_name: str = "M5_GNN_SUBSET"
    num_products: int = 16
    state_id: str = "CA"
    seed: int = 42
    max_days: int = 365
    min_nonzero_days: int = 28
    context_length: int = 28
    val_days: int = 28
    test_days: int = 28
    scaling_rule: str = "fit scaler only on train segment"
    preprocessing_rule: str = "raw daily sales, chronological order, NaN to 0 when needed"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


OFFICIAL_BENCHMARK_PROTOCOL = BenchmarkProtocol()


def sanitize_series(y: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=float)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def split_series_protocol(
    y: np.ndarray,
    val_days: int | None = None,
    test_days: int | None = None,
    min_train_days: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = sanitize_series(y)
    val_days = OFFICIAL_BENCHMARK_PROTOCOL.val_days if val_days is None else int(val_days)
    test_days = OFFICIAL_BENCHMARK_PROTOCOL.test_days if test_days is None else int(test_days)

    total = len(y)
    if total <= val_days + test_days + min_train_days:
        raise ValueError(
            f"Series length {total} is too short for val_days={val_days}, "
            f"test_days={test_days}, min_train_days={min_train_days}."
        )

    train = y[: total - val_days - test_days]
    val = y[total - val_days - test_days : total - test_days]
    test = y[total - test_days :]
    return train, val, test


def split_panel_protocol(
    panel: np.ndarray,
    val_days: int | None = None,
    test_days: int | None = None,
    min_train_days: int = 30,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(panel, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"panel must have shape (n_series, T), got {arr.shape}")

    val_days = OFFICIAL_BENCHMARK_PROTOCOL.val_days if val_days is None else int(val_days)
    test_days = OFFICIAL_BENCHMARK_PROTOCOL.test_days if test_days is None else int(test_days)
    total = arr.shape[1]

    if total <= val_days + test_days + min_train_days:
        raise ValueError(
            f"Panel length {total} is too short for val_days={val_days}, "
            f"test_days={test_days}, min_train_days={min_train_days}."
        )

    train = arr[:, : total - val_days - test_days]
    val = arr[:, total - val_days - test_days : total - test_days]
    test = arr[:, total - test_days :]
    return train, val, test


def protocol_summary_text() -> str:
    p = OFFICIAL_BENCHMARK_PROTOCOL
    return (
        f"{p.dataset_name}: max_days={p.max_days}, context_length={p.context_length}, "
        f"val_days={p.val_days}, test_days={p.test_days}, seed={p.seed}, state_id={p.state_id}"
    )
