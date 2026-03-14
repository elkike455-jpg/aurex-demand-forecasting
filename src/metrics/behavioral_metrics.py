import numpy as np


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    sa = np.std(aa)
    sb = np.std(bb)
    if sa < 1e-12 or sb < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


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


def detect_peak_rate(y_true, y_pred, prominence_ratio=0.2, tolerance_days=3):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rng = float(np.max(y_true) - np.min(y_true)) if len(y_true) else 0.0
    prominence = max(rng * prominence_ratio, 1e-8)

    peaks_true = _find_peaks_simple(y_true, prominence)
    peaks_pred = _find_peaks_simple(y_pred, prominence)

    if len(peaks_true) == 0:
        return 1.0, 0, 0

    detected = 0
    for p in peaks_true:
        if np.any(np.abs(peaks_pred - p) <= tolerance_days):
            detected += 1

    return float(detected / len(peaks_true)), int(len(peaks_true)), int(detected)


def behavioral_metrics(y_true, y_pred, trend_window=7):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    peak_rate, n_peaks, n_detected = detect_peak_rate(y_true, y_pred)

    if len(y_true) < trend_window + 1:
        trend_corr = float("nan")
    else:
        true_ma = np.convolve(y_true, np.ones(trend_window) / trend_window, mode="same")
        pred_ma = np.convolve(y_pred, np.ones(trend_window) / trend_window, mode="same")
        trend_corr = _safe_corr(np.diff(true_ma), np.diff(pred_ma))

    if len(y_true) < 2:
        direction_acc = float("nan")
    else:
        direction_acc = float((np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))).mean())

    yt_min, yt_max = np.min(y_true), np.max(y_true)
    yp_min, yp_max = np.min(y_pred), np.max(y_pred)
    y_true_norm = (y_true - yt_min) / (yt_max - yt_min + 1e-8)
    y_pred_norm = (y_pred - yp_min) / (yp_max - yp_min + 1e-8)
    shape_similarity = float(max(0.0, 1.0 - np.mean(np.abs(y_true_norm - y_pred_norm))))

    pred_std = float(np.std(y_pred))
    true_std = float(np.std(y_true))
    variance_ratio = float(pred_std / true_std) if true_std > 1e-12 else 0.0

    return {
        "peak_detection_rate": peak_rate,
        "n_peaks_real": n_peaks,
        "n_peaks_detected": n_detected,
        "trend_correlation": trend_corr,
        "direction_accuracy": direction_acc,
        "shape_similarity": shape_similarity,
        "variance_ratio": variance_ratio,
    }
