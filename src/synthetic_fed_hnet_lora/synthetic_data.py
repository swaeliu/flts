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

    train = s[:test_start]
    val = s[:test_start]
    test = s[max(0, test_start - seq_len) :]
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


def _periodic_kernel(t: np.ndarray, variance: float, period: float, lengthscale: float) -> np.ndarray:
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


def _make_regime_params(
    regime_id: int,
    variant_id: int,
    n_regimes: int,
    seed: int,
) -> Dict[str, float]:
    frac = regime_id / max(n_regimes - 1, 1)
    base = {
        "trend_slope": -0.03 + 0.06 * frac,
        "exp_trend_strength": 0.00 + 0.03 * ((regime_id % 5) / 4.0),
        "season_period_1": 12 + 36 * frac,
        "season_amp_1": 0.3 + 1.0 * ((regime_id % 7) / 6.0),
        "season_period_2": 48 + 96 * (1.0 - frac),
        "season_amp_2": 0.0 + 0.5 * ((regime_id % 3) / 2.0),
        "ar_coef_1": 0.10 + 0.45 * ((regime_id % 6) / 5.0),
        "ar_coef_2": -0.20 + 0.40 * (((regime_id + 2) % 7) / 6.0),
        "ar_coef_3": -0.10 + 0.20 * (((regime_id + 3) % 5) / 4.0),
        "noise_scale": 0.03 + 0.15 * (((regime_id * 3) % 10) / 9.0),
        "heteroskedasticity": 0.0 + 0.5 * (((regime_id * 5) % 9) / 8.0),
        "level_shift": -1.0 + 2.0 * (((regime_id * 7) % 11) / 10.0),
        "piecewise_trend": -0.02 + 0.04 * (((regime_id * 11) % 13) / 12.0),
    }

    rng = np.random.default_rng(seed + 10000 + regime_id * 101 + variant_id)
    out = dict(base)

    for k, v in base.items():
        scale = 0.10 * abs(v) + 0.01
        out[k] = float(v + rng.normal(0.0, scale))

    out["noise_scale"] = float(np.clip(out["noise_scale"], 1e-4, 5.0))
    out["heteroskedasticity"] = float(np.clip(out["heteroskedasticity"], 0.0, 2.0))
    out["exp_trend_strength"] = float(np.clip(out["exp_trend_strength"], 0.0, 0.25))
    out["season_period_1"] = float(np.clip(out["season_period_1"], 2.0, 512.0))
    out["season_period_2"] = float(np.clip(out["season_period_2"], 2.0, 1024.0))
    out["season_amp_1"] = float(np.clip(out["season_amp_1"], 0.0, 5.0))
    out["season_amp_2"] = float(np.clip(out["season_amp_2"], 0.0, 5.0))

    ar = np.array(
        [out["ar_coef_1"], out["ar_coef_2"], out["ar_coef_3"]],
        dtype=np.float64,
    )
    ar_sum = np.sum(np.abs(ar))
    if ar_sum >= 0.95:
        ar = ar * (0.95 / max(ar_sum, 1e-8))
    out["ar_coef_1"], out["ar_coef_2"], out["ar_coef_3"] = [float(a) for a in ar]

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
        out.append(_sanitize_series(mix))

    return out


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


def _estimate_window_features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    n = x.size
    if n < 4:
        return np.zeros(17, dtype=np.float32)

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
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)
    return feat


def estimate_client_feature_vector_from_series_list(
    series_list: List[np.ndarray],
    seq_len: int,
    max_windows_per_series: int = 8,
) -> torch.Tensor:

    feats: List[np.ndarray] = []

    for s in series_list:
        s = np.asarray(s, dtype=np.float32).reshape(-1)
        if s.size < max(seq_len, 8):
            continue

        if s.size <= seq_len:
            starts = [0]
        else:
            max_start = s.size - seq_len
            num = min(max_windows_per_series, max_start + 1)
            starts = np.linspace(0, max_start, num=num, dtype=int).tolist()

        for st in starts:
            win = s[st : st + seq_len]
            if win.size < seq_len:
                continue
            feats.append(_estimate_window_features(win))

    if not feats:
        return torch.zeros(17, dtype=torch.float32)

    arr = np.stack(feats, axis=0).mean(axis=0)
    return torch.from_numpy(arr).float()


def build_synthetic_client_series(cfg, seq_len: int) -> Tuple[List[Dict], torch.Tensor]:
    total_len = int(getattr(cfg, "synthetic_series_length", 4000))
    min_needed = max(
        seq_len + cfg.horizon + getattr(cfg, "synthetic_context_margin", 128),
        2 * (seq_len + cfg.horizon),
    )
    if total_len < min_needed:
        total_len = min_needed

    clients = []
    client_features = []

    n_regimes, n_variants, _ = _get_synth_shape(cfg)
    n_gp = int(getattr(cfg, "synthetic_gp_samples_per_client", 16))
    n_series_per_client = int(getattr(cfg, "synthetic_series_per_client", 10))

    client_id = 0
    for regime_id in range(n_regimes):
        for variant_id in range(n_variants):
            rng = np.random.default_rng(cfg.seed + 1000 + client_id)
            params = _make_regime_params(regime_id, variant_id, n_regimes, cfg.seed)

            base_series = [
                _sanitize_series(
                    _base_regime_series(
                        np.random.default_rng(cfg.seed + client_id * 100 + k),
                        total_len,
                        params,
                    )
                )
                for k in range(n_series_per_client)
            ]

            ks_feat_accum = {
                "kernel_linear": 0.0,
                "kernel_rbf": 0.0,
                "kernel_periodic": 0.0,
                "kernel_noise": 0.0,
            }

            if getattr(cfg, "synthetic_use_kernel_synth", True):
                gp_series = []
                for _ in range(n_gp):
                    samp, feat = sample_kernel_synth_series(
                        rng,
                        total_len,
                        int(getattr(cfg, "synthetic_kernel_terms_min", 1)),
                        int(getattr(cfg, "synthetic_kernel_terms_max", 3)),
                        float(params["season_period_1"]),
                    )
                    gp_series.append(_sanitize_series(samp))
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
            )
            client_features.append(feat_vec.numpy())
            client_id += 1

    feats = torch.from_numpy(np.stack(client_features, axis=0)).float()
    return clients, feats


def make_synthetic_clients(cfg, seq_len: int, batch_size: int):
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

    return clients, meta_rows, client_features