"""Estadística de cointegración y mean-reversion.

- engle_granger: test Engle-Granger (OLS + ADF sobre residuos)
- rolling_zscore_numba: z-score rolling compilado con numba
- half_life: half-life de reversión vía Ornstein-Uhlenbeck discreto
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numba import njit
from statsmodels.tsa.stattools import adfuller


@dataclass
class CointResult:
    """Resultado del test Engle-Granger sobre un par."""
    beta: float        # hedge ratio: y ~ alpha + beta * x
    alpha: float
    adf_stat: float
    pvalue: float
    half_life: float
    n: int


def _ols(y: np.ndarray, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    """OLS con intercepto: y = alpha + beta*x + eps."""
    n = len(y)
    sx = x.sum()
    sy = y.sum()
    sxx = (x * x).sum()
    sxy = (x * y).sum()
    denom = n * sxx - sx * sx
    if denom == 0.0:
        return 0.0, 0.0, np.zeros(n)
    beta = (n * sxy - sx * sy) / denom
    alpha = (sy - beta * sx) / n
    resid = y - alpha - beta * x
    return beta, alpha, resid


def half_life(spread: np.ndarray) -> float:
    """Half-life de reversión a la media vía OU discreto.

    Modelo: ds = -lambda * s_{t-1} dt + sigma*dW
    Estimación OLS: s_t - s_{t-1} = -lambda * s_{t-1} + eps
    Half-life = log(2) / lambda (en unidades de barra).
    """
    s = np.asarray(spread, dtype=np.float64)
    if len(s) < 3:
        return np.inf
    lag = s[:-1]
    diff = s[1:] - lag
    # OLS sin intercepto: diff = -lambda * lag
    denom = np.dot(lag, lag)
    if denom == 0.0:
        return np.inf
    lam = -np.dot(lag, diff) / denom
    if lam <= 0.0:
        return np.inf  # no reverte
    return float(np.log(2.0) / lam)


def engle_granger(y: np.ndarray, x: np.ndarray) -> CointResult:
    """Test Engle-Granger de cointegración.

    1. OLS: y = alpha + beta*x + eps
    2. ADF sobre los residuos (sin constante ni tendencia)
    3. Half-life del spread

    y, x: arrays 1D alineados (típicamente log-precios).
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = min(len(y), len(x))
    y = y[:n]
    x = x[:n]
    beta, alpha, resid = _ols(y, x)
    # ADF sobre residuos: bajo H0 el spread es I(1) -> no cointegrado.
    adf_stat, pvalue, *_ = adfuller(resid, regression="n", autolag=None, maxlag=1)
    hl = half_life(resid)
    return CointResult(
        beta=float(beta),
        alpha=float(alpha),
        adf_stat=float(adf_stat),
        pvalue=float(pvalue),
        half_life=hl,
        n=int(n),
    )


@njit(cache=True)
def rolling_mean_std(arr: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Media y desviación estándar rolling con numba.

    Salida alineada con el centro derecha: el valor en i usa datos [i-window+1, i].
    Primeras `window-1` posiciones son NaN. Ventanas con NaN se saltan (NaN out).
    """
    n = arr.shape[0]
    mean = np.full(n, np.nan)
    std = np.full(n, np.nan)
    if window <= 1 or n < window:
        return mean, std
    s = 0.0
    ss = 0.0
    nan_count = 0
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            nan_count += 1
            v = 0.0
        s += v
        ss += v * v
        if i >= window:
            old = arr[i - window]
            if np.isnan(old):
                nan_count -= 1
                old = 0.0
            s -= old
            ss -= old * old
        if i >= window - 1 and nan_count == 0:
            m = s / window
            var = ss / window - m * m
            if var < 0.0:
                var = 0.0
            mean[i] = m
            std[i] = np.sqrt(var)
    return mean, std


@njit(cache=True)
def rolling_zscore(spread: np.ndarray, window: int) -> np.ndarray:
    """Z-score del spread usando media y std rolling.

    z_i = (s_i - mean_i) / std_i
    Salida con NaN en las primeras `window-1` posiciones.
    """
    mean, std = rolling_mean_std(spread, window)
    n = spread.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        if std[i] > 0.0 and not np.isnan(std[i]):
            out[i] = (spread[i] - mean[i]) / std[i]
    return out


if __name__ == "__main__":
    # Sanity check rápido con datos sintéticos.
    rng = np.random.default_rng(0)
    n = 5000
    e = rng.normal(0, 0.01, n).cumsum()
    x = e + rng.normal(0, 0.5, n)
    y = 2.0 * e + 0.3 + rng.normal(0, 0.5, n)
    res = engle_granger(y, x)
    print(f"beta estimado: {res.beta:.4f} (esperado ~2.0)")
    print(f"alpha: {res.alpha:.4f} (esperado ~0.3)")
    print(f"ADF stat: {res.adf_stat:.4f}  pvalue: {res.pvalue:.4g}")
    print(f"half-life: {res.half_life:.2f} bars")
    z = rolling_zscore(y - res.alpha - res.beta * x, 100)
    print(f"z-score last: {z[-1]:.3f}  (mean={np.nanmean(z):.4f}, std={np.nanstd(z):.4f})")
