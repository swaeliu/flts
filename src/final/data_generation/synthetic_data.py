from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sanitize_series(x: np.ndarray, clip_value: float = 50.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=clip_value, neginf=-clip_value)
    x = np.clip(x, -clip_value, clip_value)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 0.0
    if not np.isfinite(mu):
        mu = 0.0
    if (not np.isfinite(sd)) or sd < 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    x = (x - mu) / max(sd, 1e-4)
    x = np.clip(x, -6.0, 6.0)
    return x.astype(np.float32)


def _get_synth_shape(cfg) -> Tuple[int, int, int]:
    n_regimes = int(getattr(cfg, "synthetic_num_regimes", 15))
    n_variants = int(getattr(cfg, "synthetic_variants_per_regime", 3))
    n_clients = int(getattr(cfg, "synthetic_num_clients", n_regimes * n_variants))
    if n_clients != n_regimes * n_variants:
        n_clients = n_regimes * n_variants
    return n_regimes, n_variants, n_clients


BASIC_DESCRIPTOR_FEATURE_NAMES = (
    "trend_slope",
    "exp_trend_strength",
    "season_period_1",
    "season_amp_1",
    "season_period_2",
    "season_amp_2",
    "ar_coef_1",
    "ar_coef_2",
    "ar_coef_3",
    "noise_scale",
    "heteroskedasticity",
    "level_shift",
    "piecewise_trend",
    "kernel_linear",
    "kernel_rbf",
    "kernel_periodic",
    "kernel_noise",
)

SPECTRAL_DESCRIPTOR_FEATURE_NAMES = (
    "series_std",
    "series_amplitude",
    "dominant_frequency",
    "spectral_entropy",
    "low_freq_power_ratio",
    "high_freq_power_ratio",
    "trend_strength",
    "seasonality_strength",
)


def get_descriptor_feature_names(descriptor_set: str = "basic") -> Tuple[str, ...]:
    descriptor_set = str(descriptor_set).lower()
    if descriptor_set == "basic":
        return BASIC_DESCRIPTOR_FEATURE_NAMES
    if descriptor_set == "spectral":
        return BASIC_DESCRIPTOR_FEATURE_NAMES + SPECTRAL_DESCRIPTOR_FEATURE_NAMES
    raise ValueError(f"Unknown descriptor_set: {descriptor_set}")


class SyntheticWindowDataset(Dataset):
    def __init__(
        self,
        series_list: List[np.ndarray],
        seq_len: int,
        horizon: int,
        normalize_per_series: bool = True,
        normalization_eps: float = 1e-6,
        clip_scale_min: float = 1e-4,
    ):
        self.seq_len = int(seq_len)
        self.horizon = int(horizon)
        self.normalize_per_series = bool(normalize_per_series)
        self.normalization_eps = float(normalization_eps)
        self.clip_scale_min = float(clip_scale_min)
        self.windows: List[Tuple[np.ndarray, np.ndarray, float, float]] = []

        for s in series_list:
            s = np.asarray(s, dtype=np.float32)
            if len(s) < self.seq_len + self.horizon:
                continue
            max_start = len(s) - self.seq_len - self.horizon + 1
            for st in range(max_start):
                x = s[st : st + self.seq_len]
                y = s[st + self.seq_len : st + self.seq_len + self.horizon]
                mu, sd = self._fit_norm_stats(x)
                self.windows.append((x, y, mu, sd))

    def _fit_norm_stats(self, x: np.ndarray) -> Tuple[float, float]:
        if not self.normalize_per_series:
            return 0.0, 1.0
        mu = float(np.mean(x))
        sd = float(np.std(x))
        sd = max(sd, self.clip_scale_min, self.normalization_eps)
        return mu, sd

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        x, y, mu, sd = self.windows[idx]
        if self.normalize_per_series:
            x_norm = (x - mu) / sd
            y_norm = (y - mu) / sd
        else:
            x_norm, y_norm = x, y

        mask = np.ones((self.seq_len,), dtype=np.float32)
        return (
            torch.from_numpy(x_norm[None, :]).float(),
            torch.from_numpy(y_norm[None, :]).float(),
            torch.from_numpy(mask).float(),
            torch.from_numpy(y[None, :]).float(),
            torch.tensor(mu, dtype=torch.float32),
            torch.tensor(sd, dtype=torch.float32),
        )


def split_series_gift_style(
    s: np.ndarray,
    seq_len: int,
    horizon: int,
    test_frac: float = 0.10,
):
    s = np.asarray(s, dtype=np.float32)
    n = len(s)
    if n < seq_len + horizon + 32:
        return None, None, None

    test_len = max(horizon + seq_len, int(round(test_frac * n)))
    test_start = max(seq_len + 1, n - test_len)

    # Hold out the last `horizon` points before the test region as the val
    # forecast target. Val windows may share context with train, but their
    # targets are never trained on — otherwise val loss is in-sample and
    # patience/early-stopping cannot see overfitting.
    val_start = test_start - (seq_len + horizon)
    train_end = test_start - horizon
    if val_start < 0 or train_end < seq_len + horizon:
        return None, None, None

    train = s[:train_end]
    val = s[val_start:test_start]
    test = s[max(0, test_start - seq_len):]
    return train, val, test


def compute_mase_denom_from_train_series(
    train_series: List[np.ndarray],
    seasonality: int = 1,
    eps: float = 1e-8,
) -> float:
    seasonality = max(int(seasonality), 1)
    total_abs = 0.0
    total_n = 0
    for s in train_series:
        s = np.asarray(s, dtype=np.float32)
        if len(s) <= seasonality:
            continue
        diffs = np.abs(s[seasonality:] - s[:-seasonality])
        total_abs += float(diffs.sum())
        total_n += int(diffs.size)
    if total_n == 0:
        return 1.0
    return max(total_abs / total_n, eps)


def compute_seasonal_naive_mape_from_dataset(
    dataset: SyntheticWindowDataset,
    seasonality: int = 1,
    max_batches: int | None = None,
    batch_size: int = 32,
    eps: float = 1e-8,
) -> float:
    if len(dataset) == 0:
        return 1.0

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    ape_per_series = []

    for b_idx, batch in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break

        x = batch[0].squeeze(1)
        y_true = batch[3].squeeze(1)

        m = max(int(seasonality), 1)
        H = y_true.shape[1]

        if x.shape[1] >= m:
            last_season = x[:, -m:]
            if m >= H:
                y_naive = last_season[:, -H:]
            else:
                reps = (H + m - 1) // m
                y_naive = last_season.repeat(1, reps)[:, :H]
        else:
            y_naive = x[:, -1:].repeat(1, H)

        ape = ((y_naive - y_true).abs() / (y_true.abs() + eps)).mean(dim=1)
        ape_per_series.extend(ape.tolist())

    if not ape_per_series:
        return 1.0
    return float(torch.tensor(ape_per_series).median().item())


def _linear_kernel(t: np.ndarray, variance: float) -> np.ndarray:
    return variance * np.outer(t, t)


def _rbf_kernel(t: np.ndarray, variance: float, lengthscale: float) -> np.ndarray:
    diff = t[:, None] - t[None, :]
    return variance * np.exp(-0.5 * (diff / max(lengthscale, 1e-3)) ** 2)


def _periodic_kernel(
    t: np.ndarray,
    variance: float,
    period: float,
    lengthscale: float,
) -> np.ndarray:
    diff = np.abs(t[:, None] - t[None, :])
    s = np.sin(math.pi * diff / max(period, 1e-3))
    return variance * np.exp(-2.0 * (s**2) / max(lengthscale, 1e-3) ** 2)


def _compose_random_kernel(
    rng: np.random.Generator,
    t: np.ndarray,
    terms_min: int,
    terms_max: int,
    base_period: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    n_terms = int(rng.integers(terms_min, terms_max + 1))
    parts = []
    feat = {
        "kernel_linear": 0.0,
        "kernel_rbf": 0.0,
        "kernel_periodic": 0.0,
        "kernel_noise": 0.0,
    }

    for _ in range(n_terms):
        kind = rng.choice(["linear", "rbf", "periodic"])
        if kind == "linear":
            variance = float(rng.uniform(0.02, 0.5))
            parts.append(_linear_kernel(t, variance))
            feat["kernel_linear"] += variance
        elif kind == "rbf":
            variance = float(rng.uniform(0.02, 0.5))
            lengthscale = float(rng.uniform(0.03, 0.25))
            parts.append(_rbf_kernel(t, variance, lengthscale))
            feat["kernel_rbf"] += variance
        else:
            variance = float(rng.uniform(0.02, 0.5))
            period = float(base_period * rng.uniform(0.7, 1.3))
            lengthscale = float(rng.uniform(0.05, 0.4))
            parts.append(_periodic_kernel(t, variance, period, lengthscale))
            feat["kernel_periodic"] += variance

    kernel = parts[0]
    for nxt in parts[1:]:
        op = rng.choice(["add", "mul"])
        kernel = kernel + nxt if op == "add" else kernel * (1.0 + 0.1 * nxt)

    noise = float(rng.uniform(1e-4, 2e-2))
    feat["kernel_noise"] = noise
    kernel = kernel + noise * np.eye(len(t), dtype=np.float64)
    return kernel.astype(np.float64), feat


def sample_kernel_synth_series(
    rng: np.random.Generator,
    length: int,
    terms_min: int,
    terms_max: int,
    base_period: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    t = np.linspace(0.0, 1.0, length, dtype=np.float64)
    kernel, feat = _compose_random_kernel(
        rng,
        t,
        terms_min,
        terms_max,
        max(base_period / max(length, 1), 1e-3),
    )
    sample = rng.multivariate_normal(
        mean=np.zeros(length, dtype=np.float64),
        cov=kernel,
        method="eigh",
    )
    return _sanitize_series(sample), feat


def simulate_ar_series(
    rng: np.random.Generator,
    length: int,
    coeffs: List[float],
    noise_scale: float,
    clip_value: float = 20.0,
) -> np.ndarray:
    coeffs = np.asarray(coeffs, dtype=np.float64)
    coeff_sum = np.sum(np.abs(coeffs))
    if coeff_sum >= 0.95:
        coeffs = coeffs * (0.95 / max(coeff_sum, 1e-8))

    noise_scale = max(float(noise_scale), 1e-6)

    x = np.zeros(length, dtype=np.float64)
    noise = rng.normal(0.0, noise_scale, size=length).astype(np.float64)

    for t in range(length):
        val = noise[t]
        for i, c in enumerate(coeffs, start=1):
            if t - i >= 0:
                val += float(c) * x[t - i]
        val = np.nan_to_num(val, nan=0.0, posinf=clip_value, neginf=-clip_value)
        val = np.clip(val, -clip_value, clip_value)
        x[t] = val

    return x.astype(np.float32)


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size != y.size or x.size < 2:
        return 0.0
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(x * y) / denom)


def _estimate_dominant_period(x: np.ndarray, max_period: int = 128) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 8:
        return 1.0

    x = x - x.mean()
    fft = np.fft.rfft(x)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    if power.size <= 1:
        return 1.0

    power[0] = 0.0
    valid = freqs > 0
    if not np.any(valid):
        return 1.0

    power = power[valid]
    freqs = freqs[valid]
    idx = int(np.argmax(power))
    f = float(freqs[idx])
    if f <= 1e-12:
        return 1.0

    period = 1.0 / f
    return float(np.clip(period, 1.0, max_period))


def _estimate_window_features(x: np.ndarray, descriptor_set: str = "basic") -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    feature_dim = len(get_descriptor_feature_names(descriptor_set))
    if n < 4:
        return np.zeros(feature_dim, dtype=np.float32)

    t = np.arange(n, dtype=np.float64)
    t_center = t - t.mean()
    x_center = x - x.mean()

    denom = np.sum(t_center**2)
    trend_slope = float(np.sum(t_center * x_center) / max(denom, 1e-12))

    log_abs = np.log1p(np.abs(x))
    log_abs_center = log_abs - log_abs.mean()
    exp_trend_strength = float(np.sum(t_center * log_abs_center) / max(denom, 1e-12))

    season_period_1 = _estimate_dominant_period(x, max_period=min(128, max(n - 1, 1)))
    x_detr = x - (trend_slope * t_center + x.mean())
    season_period_2 = _estimate_dominant_period(x_detr[::-1], max_period=min(256, max(n - 1, 1)))

    lag1 = int(max(1, round(season_period_1)))
    lag2 = int(max(1, round(season_period_2)))
    season_amp_1 = abs(_safe_corrcoef(x[:-lag1], x[lag1:])) if n > lag1 else 0.0
    season_amp_2 = abs(_safe_corrcoef(x[:-lag2], x[lag2:])) if n > lag2 else 0.0

    ar_coef_1 = _safe_corrcoef(x[:-1], x[1:]) if n > 1 else 0.0
    ar_coef_2 = _safe_corrcoef(x[:-2], x[2:]) if n > 2 else 0.0
    ar_coef_3 = _safe_corrcoef(x[:-3], x[3:]) if n > 3 else 0.0

    diffs = np.diff(x)
    noise_scale = float(np.std(diffs)) if diffs.size else 0.0

    resid = x - (trend_slope * t_center + x.mean())
    heteroskedasticity = abs(_safe_corrcoef(np.abs(resid), t))

    mid = n // 2
    if 0 < mid < n:
        level_shift = float(x[mid:].mean() - x[:mid].mean())
    else:
        level_shift = 0.0

    def _half_slope(z: np.ndarray) -> float:
        z = np.asarray(z, dtype=np.float64).reshape(-1)
        if z.size < 3:
            return 0.0
        tz = np.arange(z.size, dtype=np.float64)
        tz = tz - tz.mean()
        zz = z - z.mean()
        return float(np.sum(tz * zz) / max(np.sum(tz**2), 1e-12))

    piecewise_trend = _half_slope(x[mid:]) - _half_slope(x[:mid])

    trend_energy = abs(trend_slope)
    diff_var = float(np.var(diffs)) if diffs.size else 0.0
    kernel_rbf = 1.0 / (1.0 + diff_var)

    fft = np.fft.rfft(x_center)
    power = np.abs(fft) ** 2
    total_power = float(np.sum(power[1:])) if power.size > 1 else 0.0
    peak_power = float(np.max(power[1:])) if power.size > 1 else 0.0
    kernel_periodic = peak_power / max(total_power, 1e-12)
    kernel_noise = diff_var
    kernel_linear = trend_energy

    feat = np.array(
        [
            trend_slope,
            exp_trend_strength,
            season_period_1,
            season_amp_1,
            season_period_2,
            season_amp_2,
            ar_coef_1,
            ar_coef_2,
            ar_coef_3,
            noise_scale,
            heteroskedasticity,
            level_shift,
            piecewise_trend,
            kernel_linear,
            kernel_rbf,
            kernel_periodic,
            kernel_noise,
        ],
        dtype=np.float32,
    )

    if descriptor_set == "spectral":
        freqs = np.fft.rfftfreq(n, d=1.0)
        power_tail = power[1:] if power.size > 1 else np.zeros(0, dtype=np.float64)
        freq_tail = freqs[1:] if freqs.size > 1 else np.zeros(0, dtype=np.float64)
        signal_var = float(np.var(x_center))
        residual_var = float(np.var(resid))

        series_std = float(np.std(x))
        series_amplitude = float(np.percentile(x, 95) - np.percentile(x, 5))
        dominant_frequency = 1.0 / max(season_period_1, 1.0)
        seasonality_strength = kernel_periodic
        trend_strength = max(0.0, 1.0 - residual_var / max(signal_var, 1e-12))

        if power_tail.size and total_power > 1e-12:
            power_dist = power_tail / total_power
            spectral_entropy = float(
                -np.sum(power_dist * np.log(power_dist + 1e-12))
                / max(np.log(power_dist.size + 1e-12), 1e-12)
            )
            low_freq_power_ratio = float(power_tail[freq_tail <= 0.10].sum() / total_power)
            high_freq_power_ratio = float(power_tail[freq_tail >= 0.25].sum() / total_power)
        else:
            spectral_entropy = 0.0
            low_freq_power_ratio = 0.0
            high_freq_power_ratio = 0.0

        spectral_feat = np.array(
            [
                series_std,
                series_amplitude,
                dominant_frequency,
                spectral_entropy,
                low_freq_power_ratio,
                high_freq_power_ratio,
                trend_strength,
                seasonality_strength,
            ],
            dtype=np.float32,
        )
        feat = np.concatenate([feat, spectral_feat], axis=0)

    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def estimate_client_feature_vector_from_series_list(
    series_list: List[np.ndarray],
    seq_len: int,
    max_windows_per_series: int = 8,
    descriptor_set: str = "basic",
    min_window: int = 64,
) -> torch.Tensor:
    feats: List[np.ndarray] = []

    for s in series_list:
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        s = s[np.isfinite(s)]
        # Shrink the window to the series length rather than skipping short
        # series — skipping made all-short datasets collapse to an all-zero
        # descriptor and hence one shared adapter.
        if s.size < min_window:
            continue
        win_len = min(seq_len, s.size)

        if s.size <= win_len:
            starts = [0]
        else:
            max_start = s.size - win_len
            num = min(max_windows_per_series, max_start + 1)
            starts = np.linspace(0, max_start, num=num, dtype=int).tolist()

        for st in starts:
            win = s[st: st + win_len].astype(np.float64)
            # z-normalize each window so descriptors are scale-free; raw
            # windows put real (non-unit-scale) data far outside the
            # synthetic training support.
            sd = float(win.std())
            win = (win - float(win.mean())) / max(sd, 1e-8)
            feats.append(_estimate_window_features(win, descriptor_set=descriptor_set))

    if not feats:
        return torch.zeros(len(get_descriptor_feature_names(descriptor_set)), dtype=torch.float32)

    arr = np.stack(feats, axis=0).mean(axis=0)
    return torch.from_numpy(arr).float()


def _regime_separability_profile(level: str) -> Dict[str, float]:
    level = str(level).lower()
    if level == "easy":
        return {
            "center_scale": 1.6,
            "jitter_scale": 0.60,
            "contrast_scale": 1.20,
        }
    if level == "hard":
        return {
            "center_scale": 0.55,
            "jitter_scale": 1.60,
            "contrast_scale": 0.65,
        }
    return {
        "center_scale": 1.0,
        "jitter_scale": 1.0,
        "contrast_scale": 1.0,
    }


def _regime_family(regime_id: int) -> str:
    families = [
        "trend_seasonal",
        "sarima_like",
        "long_memory",
        "smooth_seasonal",
        "low_frequency",
        "multiplicative_seasonal",
        "high_frequency",
    ]
    return families[regime_id % len(families)]


def _make_regime_params(
    regime_id: int,
    variant_id: int,
    n_regimes: int,
    seed: int,
    config_variant: str = "baseline",
    separability: str = "medium",
) -> Dict[str, float]:
    frac = regime_id / max(n_regimes - 1, 1)
    family = _regime_family(regime_id)
    sep = _regime_separability_profile(separability)
    center_scale = sep["center_scale"]

    base = {
        "trend_slope": -0.03 * center_scale + 0.06 * center_scale * frac,
        "exp_trend_strength": 0.00 + 0.03 * ((regime_id % 5) / 4.0),
        "season_period_1": 12 + 36 * center_scale * frac,
        "season_amp_1": 0.3 + 1.0 * center_scale * ((regime_id % 7) / 6.0),
        "season_period_2": 48 + 96 * center_scale * (1.0 - frac),
        "season_amp_2": 0.0 + 0.5 * center_scale * ((regime_id % 3) / 2.0),
        "ar_coef_1": 0.10 + 0.45 * ((regime_id % 6) / 5.0),
        "ar_coef_2": -0.20 + 0.40 * (((regime_id + 2) % 7) / 6.0),
        "ar_coef_3": -0.10 + 0.20 * (((regime_id + 3) % 5) / 4.0),
        "noise_scale": 0.03 + 0.15 * center_scale * (((regime_id * 3) % 10) / 9.0),
        "heteroskedasticity": 0.0 + 0.5 * center_scale * (((regime_id * 5) % 9) / 8.0),
        "level_shift": -1.0 * center_scale + 2.0 * center_scale * (((regime_id * 7) % 11) / 10.0),
        "piecewise_trend": -0.02 * center_scale + 0.04 * center_scale * (((regime_id * 11) % 13) / 12.0),
        "family_id": float(regime_id % 5),
        "downsample_factor": 1.0,
        "seasonal_ar_1": 0.0,
        "seasonal_ar_2": 0.0,
        "long_memory_strength": 0.0,
        "smoothness_strength": 0.0,
    }

    rng = np.random.default_rng(seed + 10000 + regime_id * 101 + variant_id)
    out = dict(base)

    for k, v in base.items():
        if k in {"family_id", "downsample_factor"}:
            continue
        scale = (0.10 * abs(v) + 0.01) * sep["jitter_scale"]
        out[k] = float(v + rng.normal(0.0, scale))

    if family == "trend_seasonal":
        out["season_amp_1"] *= 1.25
        out["season_amp_2"] *= 1.10
        out["noise_scale"] *= 0.9

    elif family == "sarima_like":
        out["season_period_1"] = float(rng.choice([12, 24, 48, 96]))
        out["season_amp_1"] *= 1.1
        out["seasonal_ar_1"] = float(rng.uniform(0.15, 0.55))
        out["seasonal_ar_2"] = float(rng.uniform(-0.20, 0.20))
        out["ar_coef_1"] = float(rng.uniform(0.20, 0.60))
        out["ar_coef_2"] = float(rng.uniform(-0.25, 0.15))
        out["noise_scale"] *= 0.95

    elif family == "long_memory":
        out["long_memory_strength"] = float(rng.uniform(0.35, 0.85))
        out["noise_scale"] *= 0.7
        out["season_amp_1"] *= 0.8
        out["season_amp_2"] *= 0.6
        out["ar_coef_1"] = float(rng.uniform(0.15, 0.35))
        out["ar_coef_2"] = float(rng.uniform(0.05, 0.20))
        out["ar_coef_3"] = float(rng.uniform(0.00, 0.10))

    elif family == "smooth_seasonal":
        out["smoothness_strength"] = float(rng.uniform(0.7, 1.3))
        out["season_period_1"] = float(rng.choice([24, 48, 72, 96, 168]))
        out["season_period_2"] = float(rng.choice([48, 96, 168, 336]))
        out["season_amp_1"] = float(rng.uniform(0.8, 1.8))
        out["season_amp_2"] = float(rng.uniform(0.2, 1.0))
        out["noise_scale"] = float(rng.uniform(0.01, 0.05))
        out["heteroskedasticity"] *= 0.25
        out["trend_slope"] *= 0.5
        out["piecewise_trend"] *= 0.25

    elif family == "low_frequency":
        out["downsample_factor"] = float(rng.choice([2, 4, 7, 12, 24]))
        out["season_period_1"] = float(rng.choice([6, 12, 24, 52]))
        out["season_period_2"] = float(rng.choice([12, 24, 52, 104]))
        out["season_amp_1"] = float(rng.uniform(0.4, 1.4))
        out["season_amp_2"] = float(rng.uniform(0.0, 0.8))
        out["noise_scale"] *= 0.85
        out["smoothness_strength"] = float(rng.uniform(0.4, 1.0))

    elif family == "multiplicative_seasonal":
        # Log-space seasonal with level-proportional noise.
        # Mimics sales, energy demand, and web traffic where variance scales with level.
        out["season_period_1"] = float(rng.choice([4, 6, 8, 12, 24]))
        out["season_amp_1"] = float(rng.uniform(0.4, 1.2))
        out["season_period_2"] = float(rng.choice([24, 48, 52, 104]))
        out["season_amp_2"] = float(rng.uniform(0.1, 0.6))
        out["noise_scale"] = float(rng.uniform(0.03, 0.20))
        out["heteroskedasticity"] = float(rng.uniform(0.4, 1.5))
        out["exp_trend_strength"] = float(rng.uniform(0.0, 0.15))
        out["trend_slope"] *= 0.4
        out["ar_coef_1"] = float(rng.uniform(0.25, 0.65))
        out["ar_coef_2"] = float(rng.uniform(-0.20, 0.20))
        out["ar_coef_3"] = float(rng.uniform(-0.10, 0.10))
        out["level_shift"] *= 0.3
        out["piecewise_trend"] *= 0.2

    elif family == "high_frequency":
        # Short seasonal periods and strong local AR to expose the model to
        # short-cycle dynamics, improving short-horizon forecast accuracy.
        out["season_period_1"] = float(rng.choice([2, 3, 4, 6, 8]))
        out["season_amp_1"] = float(rng.uniform(0.5, 1.5))
        out["season_period_2"] = float(rng.choice([4, 6, 8, 12, 16]))
        out["season_amp_2"] = float(rng.uniform(0.2, 0.8))
        out["ar_coef_1"] = float(rng.uniform(0.50, 0.85))
        out["ar_coef_2"] = float(rng.uniform(-0.40, -0.05))
        out["ar_coef_3"] = float(rng.uniform(-0.20, 0.10))
        out["noise_scale"] = float(rng.uniform(0.05, 0.25))
        out["trend_slope"] *= 0.2
        out["exp_trend_strength"] = 0.0
        out["heteroskedasticity"] *= 0.3
        out["level_shift"] *= 0.2
        out["piecewise_trend"] *= 0.1
        out["long_memory_strength"] = 0.0

    for k, v in base.items():
        if k in {"family_id", "downsample_factor"}:
            continue
        out[k] = float(v + sep["contrast_scale"] * (out[k] - v))

    out["family"] = family
    out["noise_scale"] = float(np.clip(out["noise_scale"], 1e-4, 5.0))
    out["heteroskedasticity"] = float(np.clip(out["heteroskedasticity"], 0.0, 2.0))
    out["exp_trend_strength"] = float(np.clip(out["exp_trend_strength"], 0.0, 0.25))
    out["season_period_1"] = float(np.clip(out["season_period_1"], 2.0, 1024.0))
    out["season_period_2"] = float(np.clip(out["season_period_2"], 2.0, 2048.0))
    out["season_amp_1"] = float(np.clip(out["season_amp_1"], 0.0, 5.0))
    out["season_amp_2"] = float(np.clip(out["season_amp_2"], 0.0, 5.0))
    out["downsample_factor"] = float(np.clip(out["downsample_factor"], 1.0, 48.0))

    ar = np.array(
        [out["ar_coef_1"], out["ar_coef_2"], out["ar_coef_3"]],
        dtype=np.float64,
    )
    ar_sum = np.sum(np.abs(ar))
    if ar_sum >= 0.95:
        ar = ar * (0.95 / max(ar_sum, 1e-8))
    out["ar_coef_1"], out["ar_coef_2"], out["ar_coef_3"] = [float(a) for a in ar]

    # The sarima_like family adds seasonal AR terms (Phi1, Phi2) that the check
    # above ignores. Their combined contribution can push the effective root above
    # 1.0, causing the process to converge to a float-precision fixed point within
    # the series length (last ~20% becomes a flat constant). Cap the combined
    # non-seasonal + seasonal sum at 0.60 to keep mean-reversion fast enough.
    if out.get("family") == "sarima_like":
        s_ar_keys = ["ar_coef_1", "ar_coef_2", "ar_coef_3", "seasonal_ar_1", "seasonal_ar_2"]
        combined = sum(abs(out.get(k, 0.0)) for k in s_ar_keys)
        if combined >= 0.60:
            scale = 0.60 / combined
            for k in s_ar_keys:
                if k in out:
                    out[k] = float(out[k] * scale)

    # sarima_only forces all regimes to use the sarima_like family regardless of regime_id
    if config_variant == "sarima_only":
        out["season_period_1"] = float(rng.choice([12, 24, 48, 96]))
        out["season_amp_1"] = float(rng.uniform(0.5, 1.5))
        out["seasonal_ar_1"] = float(rng.uniform(0.15, 0.55))
        out["seasonal_ar_2"] = float(rng.uniform(-0.20, 0.20))
        out["ar_coef_1"] = float(rng.uniform(0.20, 0.60))
        out["ar_coef_2"] = float(rng.uniform(-0.25, 0.15))
        out["noise_scale"] *= 0.95
        out["family"] = "sarima_like"

    # Apply config variant modifications
    if config_variant == "low_trend":
        out["trend_slope"] = 0.0
        out["exp_trend_strength"] = 0.0
        out["piecewise_trend"] = 0.0
    elif config_variant == "high_noise":
        out["noise_scale"] = float(np.clip(out["noise_scale"] * 3.0, 1e-4, 5.0))
    elif config_variant == "high_seasonality":
        out["season_amp_1"] = float(np.clip(out["season_amp_1"] * 2.0, 0.0, 5.0))
        out["season_amp_2"] = float(np.clip(out["season_amp_2"] * 2.0, 0.0, 5.0))
    elif config_variant == "extreme_trend":
        out["trend_slope"] = 0.5
        out["exp_trend_strength"] = 0.15
        out["piecewise_trend"] = 0.2
    elif config_variant == "extreme_seasonality":
        out["season_amp_1"] = 5.0
        out["season_amp_2"] = 3.0
    elif config_variant == "extreme_noise":
        out["noise_scale"] = float(np.clip(2.0, 1e-4, 5.0))
    elif config_variant == "structural_breaks":
        out["level_shift"] = 5.0
        out["piecewise_trend"] = 0.2

    return out


def _base_regime_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    t01 = np.linspace(0.0, 1.0, length, dtype=np.float64)
    x = np.zeros(length, dtype=np.float64)

    noise_scale = max(float(params["noise_scale"]), 1e-6)
    het_strength = max(float(params["heteroskedasticity"]), 0.0)
    season_period_1 = max(float(params["season_period_1"]), 2.0)
    season_period_2 = max(float(params["season_period_2"]), 2.0)
    season_amp_1 = max(float(params["season_amp_1"]), 0.0)
    season_amp_2 = max(float(params["season_amp_2"]), 0.0)

    x += float(params["trend_slope"]) * t01 * length

    exp_part = float(params["exp_trend_strength"]) * (
        np.exp(np.clip(1.5 * t01, -2.0, 2.0)) - 1.0
    )
    x += np.clip(exp_part, -10.0, 10.0)

    x += season_amp_1 * np.sin(2.0 * np.pi * t / season_period_1)
    x += season_amp_2 * np.cos(2.0 * np.pi * t / season_period_2)

    ar = simulate_ar_series(
        rng,
        length,
        [params["ar_coef_1"], params["ar_coef_2"], params["ar_coef_3"]],
        noise_scale,
    )
    x += ar.astype(np.float64)

    cut = int(0.55 * length)
    x[cut:] += float(params["level_shift"])
    x[cut:] += float(params["piecewise_trend"]) * np.arange(length - cut, dtype=np.float64)

    het = 1.0 + het_strength * t01
    noise = rng.normal(0.0, noise_scale, size=length).astype(np.float64)
    x += noise * het

    return _sanitize_series(x)


def _simulate_sarima_like_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    p1 = max(int(round(params["season_period_1"])), 2)
    x = np.zeros(length, dtype=np.float64)
    noise_scale = max(float(params["noise_scale"]), 1e-6)

    # Do NOT multiply by length here. The AR feedback loop amplifies a large
    # trend into the clip boundary, trapping the process at a constant value for
    # the remainder of the series. Unit-range trend keeps the driving signal
    # within the seasonal amplitude, preserving diversity in the test region.
    trend = float(params["trend_slope"]) * np.linspace(0.0, 1.0, length, dtype=np.float64)
    seasonal_signal = (
        float(params["season_amp_1"]) * np.sin(2.0 * np.pi * np.arange(length) / p1)
        + float(params["season_amp_2"]) * np.cos(2.0 * np.pi * np.arange(length) / max(int(round(params["season_period_2"])), 2))
    )

    eps = rng.normal(0.0, noise_scale, size=length).astype(np.float64)
    phi1 = float(params["ar_coef_1"])
    phi2 = float(params["ar_coef_2"])
    phi3 = float(params["ar_coef_3"])
    Phi1 = float(params.get("seasonal_ar_1", 0.0))
    Phi2 = float(params.get("seasonal_ar_2", 0.0))

    for t in range(length):
        val = trend[t] + seasonal_signal[t] + eps[t]
        if t - 1 >= 0:
            val += phi1 * x[t - 1]
        if t - 2 >= 0:
            val += phi2 * x[t - 2]
        if t - 3 >= 0:
            val += phi3 * x[t - 3]
        if t - p1 >= 0:
            val += Phi1 * x[t - p1]
        if t - 2 * p1 >= 0:
            val += Phi2 * x[t - 2 * p1]
        x[t] = np.clip(val, -20.0, 20.0)

    return _sanitize_series(x)


def _simulate_long_memory_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    d = float(np.clip(params.get("long_memory_strength", 0.5), 0.0, 0.95))
    noise_scale = max(float(params["noise_scale"]), 1e-6)

    eps = rng.normal(0.0, noise_scale, size=length + 256).astype(np.float64)
    max_lag = min(128, length)
    weights = np.array([(k + 1) ** (-d) for k in range(max_lag)], dtype=np.float64)
    weights = weights / max(weights.sum(), 1e-12)

    x = np.zeros(length, dtype=np.float64)
    base = _base_regime_series(rng, length, params).astype(np.float64)

    for t in range(length):
        mem = 0.0
        for k in range(min(t + 1, max_lag)):
            mem += weights[k] * eps[t - k + 128]
        x[t] = base[t] + mem

    return _sanitize_series(x)


def _moving_average_smooth(x: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 1)
    if width <= 1:
        return np.asarray(x, dtype=np.float64)
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(np.asarray(x, dtype=np.float64), kernel, mode="same")


def _simulate_smooth_seasonal_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    p1 = max(float(params["season_period_1"]), 2.0)
    p2 = max(float(params["season_period_2"]), 2.0)
    smoothness = max(float(params.get("smoothness_strength", 1.0)), 0.1)

    x = (
        float(params["season_amp_1"]) * np.sin(2.0 * np.pi * t / p1)
        + float(params["season_amp_2"]) * np.cos(2.0 * np.pi * t / p2)
        + float(params["trend_slope"]) * np.linspace(0.0, 1.0, length, dtype=np.float64) * length * 0.5
    )

    noise = rng.normal(0.0, max(float(params["noise_scale"]), 1e-6), size=length).astype(np.float64)
    x = x + noise
    width = int(max(3, round(3 + 8 * smoothness)))
    x = _moving_average_smooth(x, width)
    return _sanitize_series(x)


def _downsample_series(
    x: np.ndarray,
    factor: int,
    target_length: int,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    factor = max(int(factor), 1)
    if factor == 1:
        y = x
    else:
        usable = (x.size // factor) * factor
        if usable < factor:
            y = x
        else:
            y = x[:usable].reshape(-1, factor).mean(axis=1)

    if y.size < target_length:
        reps = (target_length + y.size - 1) // max(y.size, 1)
        y = np.tile(y, reps)[:target_length]
    else:
        y = y[:target_length]
    return _sanitize_series(y)


def _simulate_low_frequency_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    factor = int(round(params.get("downsample_factor", 4.0)))
    fine_length = max(length * factor, length + 8)

    base = _simulate_smooth_seasonal_series(rng, fine_length, params).astype(np.float64)
    drift = float(params["trend_slope"]) * np.linspace(0.0, 1.0, fine_length, dtype=np.float64) * fine_length * 0.25
    base = base + drift
    return _downsample_series(base, factor=factor, target_length=length)


def _simulate_multiplicative_seasonal_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    p1 = max(float(params["season_period_1"]), 2.0)
    p2 = max(float(params["season_period_2"]), 2.0)
    noise_scale = max(float(params["noise_scale"]), 1e-6)

    # Build signal in log space so seasonality and trend interact multiplicatively.
    log_trend = (
        float(params["trend_slope"]) * np.linspace(0.0, 1.0, length, dtype=np.float64) * 2.0
        + float(params["exp_trend_strength"]) * np.linspace(0.0, 1.0, length, dtype=np.float64)
    )
    log_seasonal = (
        float(params["season_amp_1"]) * np.sin(2.0 * np.pi * t / p1)
        + float(params["season_amp_2"]) * np.cos(2.0 * np.pi * t / p2)
    )
    ar_noise = simulate_ar_series(
        rng, length,
        [params["ar_coef_1"], params["ar_coef_2"], params["ar_coef_3"]],
        noise_scale * 0.5,
    )
    level = np.exp(np.clip(log_trend + log_seasonal + 0.4 * ar_noise.astype(np.float64), -4.0, 4.0))

    # Proportional noise: variance scales with current level.
    het = max(float(params["heteroskedasticity"]), 0.0)
    prop_noise = rng.normal(0.0, noise_scale, size=length).astype(np.float64)
    x = level * (1.0 + het * prop_noise)
    return _sanitize_series(x)


def _simulate_high_frequency_series(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    p1 = max(float(params["season_period_1"]), 2.0)
    p2 = max(float(params["season_period_2"]), 2.0)
    noise_scale = max(float(params["noise_scale"]), 1e-6)

    # Short-period seasonal baseline.
    x = (
        float(params["season_amp_1"]) * np.sin(2.0 * np.pi * t / p1)
        + float(params["season_amp_2"]) * np.sin(2.0 * np.pi * t / p2 + np.pi / 3.0)
    )
    # Strong local AR — high autocorrelation at lag 1 teaches the model short-range dynamics.
    ar_component = simulate_ar_series(
        rng, length,
        [params["ar_coef_1"], params["ar_coef_2"], params["ar_coef_3"]],
        noise_scale,
    )
    x += ar_component.astype(np.float64)
    x += float(params["trend_slope"]) * np.linspace(0.0, 1.0, length, dtype=np.float64) * length * 0.2
    return _sanitize_series(x)


def _generate_series_for_family(
    rng: np.random.Generator,
    length: int,
    params: Dict[str, float],
) -> np.ndarray:
    family = params["family"]

    if family == "trend_seasonal":
        return _base_regime_series(rng, length, params)
    if family == "sarima_like":
        return _simulate_sarima_like_series(rng, length, params)
    if family == "long_memory":
        return _simulate_long_memory_series(rng, length, params)
    if family == "smooth_seasonal":
        return _simulate_smooth_seasonal_series(rng, length, params)
    if family == "low_frequency":
        return _simulate_low_frequency_series(rng, length, params)
    if family == "multiplicative_seasonal":
        return _simulate_multiplicative_seasonal_series(rng, length, params)
    if family == "high_frequency":
        return _simulate_high_frequency_series(rng, length, params)

    return _base_regime_series(rng, length, params)


def tsmixup(
    rng: np.random.Generator,
    base_series: List[np.ndarray],
    out_count: int,
    max_k: int = 3,
    alpha: float = 0.7,
) -> List[np.ndarray]:
    if not base_series:
        return []

    series_len = len(base_series[0])
    out = []

    for _ in range(out_count):
        k = int(rng.integers(1, max_k + 1))
        idx = rng.choice(len(base_series), size=k, replace=True)
        lam = rng.dirichlet(alpha=np.full(k, alpha, dtype=np.float64))
        mix = np.zeros(series_len, dtype=np.float64)
        for w, i in zip(lam, idx):
            s = _sanitize_series(base_series[int(i)])
            mix += float(w) * s.astype(np.float64)
        out.append(_sanitize_series(mix[:series_len]))

    return out


def build_synthetic_client_series(cfg, seq_len: int) -> Tuple[List[Dict], torch.Tensor]:
    total_len = int(getattr(cfg, "synthetic_series_length", 4000))

    long_horizon_factor = int(getattr(cfg, "synthetic_long_horizon_factor", 4))
    context_margin = int(getattr(cfg, "synthetic_context_margin", 128))

    min_needed = max(
        seq_len + cfg.horizon + context_margin,
        2 * (seq_len + cfg.horizon),
        long_horizon_factor * (seq_len + cfg.horizon),
    )
    if total_len < min_needed:
        total_len = min_needed

    clients = []
    client_features = []

    n_regimes, n_variants, _ = _get_synth_shape(cfg)
    descriptor_set = str(getattr(cfg, "descriptor_set", "basic")).lower()
    use_oracle_regime_id = bool(getattr(cfg, "use_oracle_regime_id", False))
    separability = str(getattr(cfg, "synthetic_regime_separability", "medium")).lower()
    feature_names = list(get_descriptor_feature_names(descriptor_set))
    if use_oracle_regime_id:
        feature_names.extend([f"oracle_regime_{i:02d}" for i in range(n_regimes)])
    cfg.synthetic_feature_names = tuple(feature_names)

    n_gp = int(getattr(cfg, "synthetic_gp_samples_per_client", 16))
    n_series_per_client = int(getattr(cfg, "synthetic_series_per_client", 10))

    client_id = 0
    for regime_id in range(n_regimes):
        for variant_id in range(n_variants):
            rng = np.random.default_rng(cfg.seed + 1000 + client_id)
            config_variant = getattr(cfg, "synthetic_config_variant", "baseline")
            params = _make_regime_params(
                regime_id,
                variant_id,
                n_regimes,
                cfg.seed,
                config_variant,
                separability=separability,
            )

            base_series = []
            for k in range(n_series_per_client):
                srng = np.random.default_rng(cfg.seed + client_id * 100 + k)
                s = _generate_series_for_family(srng, total_len, params)
                base_series.append(_sanitize_series(s))

            ks_feat_accum = {
                "kernel_linear": 0.0,
                "kernel_rbf": 0.0,
                "kernel_periodic": 0.0,
                "kernel_noise": 0.0,
            }

            if getattr(cfg, "synthetic_use_kernel_synth", True):
                gp_series = []
                gp_weight = float(getattr(cfg, "synthetic_kernel_blend_weight", 0.35))
                for _ in range(n_gp):
                    samp, feat = sample_kernel_synth_series(
                        rng,
                        total_len,
                        int(getattr(cfg, "synthetic_kernel_terms_min", 1)),
                        int(getattr(cfg, "synthetic_kernel_terms_max", 3)),
                        float(params["season_period_1"]),
                    )

                    if params["family"] in {"smooth_seasonal", "low_frequency"}:
                        width = int(max(3, round(params.get("downsample_factor", 1.0) + 2)))
                        samp = _moving_average_smooth(samp, width)

                    blended = (1.0 - gp_weight) * samp + gp_weight * base_series[int(rng.integers(0, len(base_series)))]
                    gp_series.append(_sanitize_series(blended))

                    for k, v in feat.items():
                        ks_feat_accum[k] += float(v)

                base_series.extend(gp_series)

            if getattr(cfg, "synthetic_use_mixup", True):
                base_series.extend(
                    tsmixup(
                        rng,
                        base_series,
                        int(getattr(cfg, "synthetic_mixup_per_client", 8)),
                        int(getattr(cfg, "synthetic_max_mix_components", 3)),
                        float(getattr(cfg, "synthetic_mixup_alpha", 0.7)),
                    )
                )

            meta = {
                "client_id": client_id,
                "regime_id": regime_id,
                "variant_id": variant_id,
                "dataset": "synthetic",
                "regime": f"regime_{regime_id:02d}",
                "regime_variant": f"regime_{regime_id:02d}_variant_{variant_id:02d}",
                "freq": "synthetic",
                "family": params["family"],
                "descriptor_set": descriptor_set,
                "synthetic_regime_separability": separability,
                "use_oracle_regime_id": use_oracle_regime_id,
                **params,
                **ks_feat_accum,
                "n_raw_series": len(base_series),
            }
            clients.append({"series": base_series, "meta": meta})

            feat_vec = estimate_client_feature_vector_from_series_list(
                base_series,
                seq_len=seq_len,
                max_windows_per_series=int(
                    getattr(cfg, "client_feature_windows_per_series", 8)
                ),
                descriptor_set=descriptor_set,
            )
            if use_oracle_regime_id:
                oracle_feat = torch.zeros(n_regimes, dtype=torch.float32)
                oracle_feat[regime_id] = 1.0
                feat_vec = torch.cat([feat_vec, oracle_feat], dim=0)
            client_features.append(feat_vec.numpy())
            client_id += 1

    feats = torch.from_numpy(np.stack(client_features, axis=0)).float()
    return clients, feats


def make_synthetic_clients(cfg, seq_len: int, batch_size: int, return_feature_stats: bool = False):
    raw_clients, client_features = build_synthetic_client_series(cfg, seq_len=seq_len)

    clients = []
    meta_rows = []
    kept_feature_rows = []

    for raw, feat_row in zip(raw_clients, client_features):
        train_series, val_series, test_series = [], [], []
        for s in raw["series"]:
            tr, va, te = split_series_gift_style(
                s,
                seq_len=seq_len,
                horizon=cfg.horizon,
                test_frac=cfg.test_frac,
            )
            if tr is None:
                continue
            train_series.append(tr)
            val_series.append(va)
            test_series.append(te)

        train_ds = SyntheticWindowDataset(
            train_series,
            seq_len=seq_len,
            horizon=cfg.horizon,
            normalize_per_series=cfg.normalize_per_series,
            normalization_eps=cfg.normalization_eps,
            clip_scale_min=cfg.clip_scale_min,
        )
        val_ds = SyntheticWindowDataset(
            val_series,
            seq_len=seq_len,
            horizon=cfg.horizon,
            normalize_per_series=cfg.normalize_per_series,
            normalization_eps=cfg.normalization_eps,
            clip_scale_min=cfg.clip_scale_min,
        )
        test_ds = SyntheticWindowDataset(
            test_series,
            seq_len=seq_len,
            horizon=cfg.horizon,
            normalize_per_series=cfg.normalize_per_series,
            normalization_eps=cfg.normalization_eps,
            clip_scale_min=cfg.clip_scale_min,
        )

        if len(train_ds) == 0 or len(val_ds) == 0 or len(test_ds) == 0:
            print(
                f"[warn] skipping client {raw['meta']['client_id']} "
                f"({raw['meta']['regime_variant']}): "
                f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
            )
            continue

        clients.append(
            {
                "train": DataLoader(
                    train_ds,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=cfg.num_workers,
                ),
                "val": DataLoader(
                    val_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                ),
                "test": DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                ),
                "train_series": train_series,
                "mase_denom": compute_mase_denom_from_train_series(
                    train_series,
                    seasonality=getattr(cfg, "mase_seasonality", 1),
                ),
                "seasonal_naive_mape": compute_seasonal_naive_mape_from_dataset(
                    test_ds,
                    seasonality=getattr(cfg, "mase_seasonality", 1),
                    max_batches=cfg.eval_batches,
                    batch_size=batch_size,
                ),
            }
        )
        meta_rows.append(raw["meta"])
        kept_feature_rows.append(feat_row)

    if not kept_feature_rows:
        raise RuntimeError(
            "No valid synthetic clients were constructed. "
            "Increase synthetic_series_length or reduce seq_len/horizon."
        )

    client_features = torch.stack(kept_feature_rows, dim=0)
    if getattr(cfg, "normalize_client_features", True):
        mu = client_features.mean(dim=0, keepdim=True)
        sd = client_features.std(dim=0, keepdim=True).clamp_min(1e-6)
        client_features = (client_features - mu) / sd
    else:
        mu = torch.zeros(1, client_features.shape[1])
        sd = torch.ones(1, client_features.shape[1])

    if return_feature_stats:
        # Raw-feature stats, saved as client_feature_stats.pt so eval scripts
        # can z-score out-of-distribution inputs the same way as training.
        # The fallback (recomputing stats from the *normalized* features in
        # server.pt) yields ~(0, 1) and leaves eval inputs unnormalized.
        feature_stats = {"mean": mu.squeeze(0).clone(), "std": sd.squeeze(0).clone()}
        return clients, meta_rows, client_features, feature_stats
    return clients, meta_rows, client_features
