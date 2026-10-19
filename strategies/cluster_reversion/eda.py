"""EDA cluster reversion: verificar que el residual de la regresión
alt vs BTC revierte (autocorrelación negativa significativa).

Si el z-score del residual tiene AC1 < 0 y significativo, hay mean reversion
estadística explotable. Si AC1 ~ 0 o positivo, no hay reversión y descartamos.

Modelo:
  - Beta rolling: ret_alt = alpha + beta * ret_BTC + eps (ventana 288 = 24h)
  - Residual acumulado: suma de eps en últimas K velas (K=6 = 30 min)
  - Z-score: (residual - mean_residual_rolling) / std_residual_rolling

Uso:
    python -m strategies.cluster_reversion.eda
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from src.data_loader import _cache_path
from src.universe_lead_lag import LEAD_LAG_UNIVERSE, LEADER

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMEFRAME = "5m"

BETA_WINDOW = 288       # 24h en velas 5m
RESID_K = 6             # acumular 30 min de residual
Z_WINDOW = 288          # ventana z-score


def load_returns(symbol: str) -> pl.DataFrame:
    cache = _cache_path(symbol, TIMEFRAME, "perp")
    df = pl.read_parquet(cache).select("open_time", "close").sort("open_time")
    df = df.with_columns(
        pl.col("close").log().diff().alias("ret"),
    )
    return df.drop_nulls()


@njit(cache=True, fastmath=True)
def rolling_beta(ret_y: np.ndarray, ret_x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Beta y alpha rolling: y = alpha + beta*x.

    Salida alineada al centro derecho. NaN en primeras window-1.
    """
    n = len(ret_y)
    beta = np.full(n, np.nan)
    alpha = np.full(n, np.nan)
    if window <= 1 or n < window:
        return beta, alpha
    sx = 0.0
    sy = 0.0
    sxx = 0.0
    sxy = 0.0
    for i in range(n):
        x = ret_x[i]
        y = ret_y[i]
        if np.isnan(x) or np.isnan(y):
            x = 0.0
            y = 0.0
        sx += x
        sy += y
        sxx += x * x
        sxy += x * y
        if i >= window:
            ox = ret_x[i - window]
            oy = ret_y[i - window]
            if np.isnan(ox):
                ox = 0.0
            if np.isnan(oy):
                oy = 0.0
            sx -= ox
            sy -= oy
            sxx -= ox * ox
            sxy -= ox * oy
        if i >= window - 1:
            denom = window * sxx - sx * sx
            if denom != 0.0:
                b = (window * sxy - sx * sy) / denom
                a = (sy - b * sx) / window
                beta[i] = b
                alpha[i] = a
    return beta, alpha


@njit(cache=True)
def rolling_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Suma rolling tratando NaN como 0. Salida alineada al centro derecha."""
    n = len(arr)
    out = np.full(n, np.nan)
    if window <= 0 or n < window:
        return out
    s = 0.0
    for i in range(n):
        v = arr[i]
        if np.isnan(v):
            v = 0.0
        s += v
        if i >= window:
            old = arr[i - window]
            if np.isnan(old):
                old = 0.0
            s -= old
        if i >= window - 1:
            out[i] = s
    return out


def residual_zscore(alt_ret: np.ndarray, btc_ret: np.ndarray) -> np.ndarray:
    """Calcula el z-score del residual acumulado."""
    beta, alpha = rolling_beta(alt_ret, btc_ret, BETA_WINDOW)
    # Residual instantáneo.
    eps = alt_ret - alpha - beta * btc_ret
    # Residual acumulado en K velas.
    eps_cum = rolling_sum(eps, RESID_K)
    # Z-score del residual acumulado.
    from src.stats import rolling_zscore
    z = rolling_zscore(eps_cum, Z_WINDOW)
    return z


def ac1(arr: np.ndarray) -> float:
    """Autocorrelación lag-1, ignorando NaN."""
    mask = ~np.isnan(arr)
    a = arr[mask]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def half_life_z(z: np.ndarray) -> float:
    """Half-life de reversión del z-score."""
    mask = ~np.isnan(z)
    a = z[mask]
    if len(a) < 3:
        return float("inf")
    lag = a[:-1]
    diff = a[1:] - lag
    denom = np.dot(lag, lag)
    if denom == 0:
        return float("inf")
    lam = -np.dot(lag, diff) / denom
    if lam <= 0:
        return float("inf")
    return float(np.log(2.0) / lam)


def main() -> int:
    print(f"=== EDA cluster reversion, timeframe {TIMEFRAME} ===")
    print(f"Params: beta_window={BETA_WINDOW}  resid_k={RESID_K}  z_window={Z_WINDOW}")
    print(f"BTC = {LEADER}\n")
    btc_df = load_returns(LEADER)
    btc_ret = btc_df["ret"].to_numpy()

    rows = []
    for alt in LEAD_LAG_UNIVERSE:
        alt_df = load_returns(alt)
        joined = btc_df.drop("ret").join(
            alt_df, on="open_time", how="inner", suffix="_alt"
        ).sort("open_time")
        # Recalcular ret alineados.
        btc_r = np.log(joined["close"].to_numpy())
        alt_r = np.log(joined["close_alt"].to_numpy())
        btc_ret_a = np.diff(btc_r)
        alt_ret_a = np.diff(alt_r)
        z = residual_zscore(alt_ret_a, btc_ret_a)
        ac = ac1(z)
        hl = half_life_z(z)
        # Cuántas veces |z| > 2.
        mask = ~np.isnan(z)
        extremes = np.sum(np.abs(z[mask]) > 2.0)
        pct_extreme = extremes / mask.sum() * 100
        # Reversión post-extremo: si z > 2 en t, ¿z baja en t+1?
        z_clean = z[mask]
        above = z_clean[:-1] > 2.0
        below = z_clean[:-1] < -2.0
        reversion_up = np.mean(z_clean[1:][above] < z_clean[:-1][above]) if above.sum() > 0 else float("nan")
        reversion_down = np.mean(z_clean[1:][below] > z_clean[:-1][below]) if below.sum() > 0 else float("nan")
        rows.append({
            "alt": alt,
            "n": int(mask.sum()),
            "ac1": round(ac, 4),
            "half_life": round(hl, 2),
            "pct_extreme": round(float(pct_extreme), 2),
            "n_extreme": int(extremes),
            "reversion_when_high": round(float(reversion_up), 3) if not np.isnan(reversion_up) else None,
            "reversion_when_low": round(float(reversion_down), 3) if not np.isnan(reversion_down) else None,
        })
        print(
            f"{alt:10s}  ac1={ac:+.4f}  HL={hl:6.2f}  "
            f"|z|>2: {extremes:5d} ({pct_extreme:4.1f}%)  "
            f"rev_high={reversion_up:.3f}  rev_low={reversion_down:.3f}"
        )
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_DIR / "reversion_eda.csv")
    print(f"\nGuardado: {RESULTS_DIR}/reversion_eda.csv")
    # Veredicto.
    sig_reversion = [r for r in rows if r["ac1"] < -0.02]
    print(f"\nAlts con AC1 < -0.02 (reversión detectada): {len(sig_reversion)}/{len(rows)}")
    rev_high = [r["reversion_when_high"] for r in rows if r["reversion_when_high"] is not None]
    rev_low = [r["reversion_when_low"] for r in rows if r["reversion_when_low"] is not None]
    if rev_high:
        print(f"Reversión promedio cuando z>2: {np.mean(rev_high):.1%} (probabilidad de bajar)")
    if rev_low:
        print(f"Reversión promedio cuando z<-2: {np.mean(rev_low):.1%} (probabilidad de subir)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
