"""Cálculo de vol realizada (RV) a partir de klines 1h de Binance.

RV anualizada usando log-returns, ventana móvil de N horas.
Comparamos con DVOL (IV 30d) para calcular el Vol Risk Premium (VRP).

Uso:
    python -m strategies.derivatives.rv_calculator
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from src.data_loader import load_klines

CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-10-09"
END = "2026-07-04"


@njit(cache=True)
def realized_vol_window(log_ret: np.ndarray, window: int, periods_per_year: int = 365 * 24) -> np.ndarray:
    """Vol realizada anualizada usando ventana móvil.

    RV_t = std(log_ret[t-window:t]) * sqrt(periods_per_year)
    Salida alineada al centro derecha. NaN en primeras window-1.
    """
    n = log_ret.shape[0]
    out = np.full(n, np.nan)
    if window <= 1 or n < window:
        return out
    s = 0.0
    ss = 0.0
    for i in range(n):
        v = log_ret[i]
        if np.isnan(v):
            v = 0.0
        s += v
        ss += v * v
        if i >= window:
            old = log_ret[i - window]
            if np.isnan(old):
                old = 0.0
            s -= old
            ss -= old * old
        if i >= window - 1:
            m = s / window
            var = ss / window - m * m
            if var < 0.0:
                var = 0.0
            out[i] = np.sqrt(var) * np.sqrt(periods_per_year) * 100.0  # en % anualizado
    return out


def compute_rv(symbol: str, window_hours: int = 720) -> pl.DataFrame:
    """Calcula RV diaria para un símbolo usando klines 1h.

    window_hours: ventana para calcular RV (720h = 30 días, comparable a DVOL 30d).
    Devuelve DataFrame con: date, rv_close (RV anualizada en %).
    """
    df = load_klines(symbol, "1h", START, END, market="perp")
    df = df.with_columns(
        pl.col("close").log().diff().alias("log_ret"),
    )
    log_ret = df["log_ret"].to_numpy()
    rv = realized_vol_window(log_ret, window_hours)
    df = df.with_columns(pl.Series(name="rv", values=rv))
    # Agregar a diario tomando el último valor de cada día.
    df = df.with_columns(pl.col("open_time").dt.date().alias("date"))
    daily = (
        df.group_by("date")
        .agg(pl.col("rv").last().alias("rv_close"))
        .sort("date")
    )
    return daily


def main() -> int:
    for sym in ["BTCUSDT", "ETHUSDT"]:
        rv = compute_rv(sym, window_hours=720)
        out = CACHE_DIR / f"RV_{sym}_30d.csv"
        rv.write_csv(out)
        valid = rv.filter(pl.col("rv_close").is_not_null())
        print(f"{sym}: {valid.height} días con RV  rango {valid['rv_close'].min():.2f}-{valid['rv_close'].max():.2f}%")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
