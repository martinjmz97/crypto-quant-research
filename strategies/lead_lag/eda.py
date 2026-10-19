"""EDA lead-lag: cross-correlation function entre BTC y cada alt.

Para cada alt calcula la CCF del retorno de BTC contra el retorno de la alt
a distintos lags (en barras de 5m). Si BTC lidera, la correlación será
máxima en lag > 0 (BTC de N barras atrás predice alt actual).

También reporta:
- Correlación contemporánea (lag 0)
- Lag con máxima correlación
- Correlación a lags 1, 2, 3 (1-3 barras = 5-15 min)
- Significancia: intervalos de confianza via bootstrap

Uso:
    python -m strategies.lead_lag.eda
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import _cache_path
from src.universe_lead_lag import LEAD_LAG_UNIVERSE, LEADER

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIMEFRAME = "5m"


def load_close(symbol: str) -> pl.DataFrame:
    cache = _cache_path(symbol, TIMEFRAME, "perp")
    return pl.read_parquet(cache).select("open_time", "close")


def log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(close))


def cross_corr(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """CCF normalizada: corr(x[t-lag], y[t]) para lag en [0, max_lag].

    x = BTC (líder hipotético), y = alt (seguidor hipotético).
    Lag > 0 significa: BTC de `lag` barras atrás contra alt actual.
    """
    n = min(len(x), len(y))
    x = x[-n:]
    y = y[-n:]
    x = (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
    y = (y - y.mean()) / (y.std() if y.std() > 0 else 1.0)
    ccf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        if lag == 0:
            ccf[lag] = np.mean(x * y)
        else:
            ccf[lag] = np.mean(x[:-lag] * y[lag:])
    return ccf


def bootstrap_ccf_ci(
    x: np.ndarray, y: np.ndarray, max_lag: int,
    n_boot: int = 500, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """IC 95% para la CCF via bootstrap por bloques (block size = max_lag+10)."""
    rng = np.random.default_rng(seed)
    n = min(len(x), len(y))
    block = max_lag + 10
    n_blocks = n // block
    boots = np.zeros((n_boot, max_lag + 1))
    for b in range(n_boot):
        idx = rng.integers(0, n_blocks, size=n_blocks)
        starts = idx * block
        # Construir series resampleadas por bloques.
        xs = np.concatenate([x[s : s + block] for s in starts])
        ys = np.concatenate([y[s : s + block] for s in starts])
        m = min(len(xs), len(ys))
        boots[b] = cross_corr(xs[:m], ys[:m], max_lag)
    lo = np.percentile(boots, 2.5, axis=0)
    hi = np.percentile(boots, 97.5, axis=0)
    return lo, hi


def main() -> int:
    print(f"=== EDA lead-lag {LEADER}→alts, timeframe {TIMEFRAME} ===\n")
    btc = load_close(LEADER)
    btc_ret = log_returns(btc["close"].to_numpy())

    rows = []
    max_lag = 12  # 12 barras * 5m = 60 min máximo rezago analizado
    for alt in LEAD_LAG_UNIVERSE:
        alt_df = load_close(alt)
        # Alinear por open_time (inner join).
        joined = btc.join(alt_df, on="open_time", how="inner", suffix="_alt").sort("open_time")
        btc_c = joined["close"].to_numpy()
        alt_c = joined["close_alt"].to_numpy()
        btc_r = log_returns(btc_c)
        alt_r = log_returns(alt_c)
        ccf = cross_corr(btc_r, alt_r, max_lag)
        lo, hi = bootstrap_ccf_ci(btc_r, alt_r, max_lag)
        best_lag = int(np.argmax(ccf))
        sig = ccf > hi  # significancia 95% bootstrap
        rows.append({
            "alt": alt,
            "n": len(btc_r),
            "ccf_0": round(ccf[0], 4),
            "ccf_1": round(ccf[1], 4),
            "ccf_2": round(ccf[2], 4),
            "ccf_3": round(ccf[3], 4),
            "best_lag": best_lag,
            "best_ccf": round(ccf[best_lag], 4),
            "sig_lag1": bool(sig[1]),
            "sig_lag2": bool(sig[2]),
            "sig_lag3": bool(sig[3]),
        })
        print(
            f"{alt:10s}  ccf0={ccf[0]:+.4f}  ccf1={ccf[1]:+.4f}({'*' if sig[1] else ' '})  "
            f"ccf2={ccf[2]:+.4f}({'*' if sig[2] else ' '})  ccf3={ccf[3]:+.4f}({'*' if sig[3] else ' '})  "
            f"best_lag={best_lag} ({best_lag*5}min)  best_ccf={ccf[best_lag]:+.4f}"
        )
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_DIR / "lead_lag_ccf.csv")
    print(f"\nGuardado: {RESULTS_DIR}/lead_lag_ccf.csv")
    print(f"\n* = significativo al 95% (bootstrap)")
    # Resumen: cuántas alts tienen lead-lag significativo en lag 1-3.
    sig_count = sum(1 for r in rows if r["sig_lag1"] or r["sig_lag2"] or r["sig_lag3"])
    print(f"\nAlts con lead-lag significativo (lag 1-3): {sig_count}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
