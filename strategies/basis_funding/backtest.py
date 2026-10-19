"""Backtest de basis funding arb.

Estrategia:
  - Long spot + short perp (delta-neutral) cuando funding > umbral.
  - Entrada: funding_rate > percentil 80 de ventana rolling (lookback).
  - Salida: funding_rate < percentil 50 (mediana rolling) o max hold.
  - Costos: round-trip taker = 2 * 0.05% por lado = 20 bps total.
  - PnL = funding cobrado + cambio en base premium - costos.

Reglas:
  - Si position_abierta y funding > 0: cobrar funding cada 8h (al funding_time).
  - Salir cuando funding < mediana rolling o alcanzado max_hold_intervals.
  - Delta-neutral: notional $1 long spot + $1 short perp.
  - PnL del basis = (perp_t - perp_0) * (-1) + (spot_t - spot_0) * (+1)
                   = -(perp_t - perp_0) + (spot_t - spot_0)
    (short perp gana si perp cae; long spot gana si spot sube)
  - Pero lo que cobramos es funding (positivo = longs pagan a shorts).

Uso:
    python -m strategies.basis_funding.backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import _cache_path, load_funding, load_klines
from src.metrics import summary
from src.universe import UNIVERSE

RESULTS_DIR = Path(__file__).resolve().path.parent / "results" if False else Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-01-01"
END = "2024-12-31"

# Parámetros (no más de 5).
LOOKBACK = 90          # ventana rolling para percentiles (90 intervalos = 30 días)
ENTRY_PCT = 0.80       # entrar si funding > percentil 80
EXIT_PCT = 0.50        # salir si funding < mediana
MAX_HOLD = 60          # máximo intervalos en posición (60 * 8h = 20 días)
FEE_TAKER = 0.0005     # 0.05% taker por lado
ROUND_TRIP_COST = 4 * FEE_TAKER  # entrada+salida, spot+perp = 4 fills


def load_aligned(symbol: str) -> pl.DataFrame:
    """Carga funding + spot close + perp close alineados por funding_time."""
    fund = load_funding(symbol, START, END).select(
        "funding_time", "funding_rate", "mark_price"
    )
    spot = pl.read_parquet(_cache_path(symbol, "1h", "spot")).select(
        pl.col("open_time"), pl.col("close").alias("spot_close")
    )
    perp = pl.read_parquet(_cache_path(symbol, "1h", "perp")).select(
        pl.col("open_time"), pl.col("close").alias("perp_close")
    )
    df = (
        fund.join(spot, left_on="funding_time", right_on="open_time", how="inner")
        .join(perp, left_on="funding_time", right_on="open_time", how="inner")
        .sort("funding_time")
    )
    return df


def rolling_quantile(arr: np.ndarray, window: int, q: float) -> np.ndarray:
    """Percentil rolling. Salida alineada al centro derecho."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) > 0:
            out[i] = np.quantile(valid, q)
    return out


def backtest_symbol(symbol: str) -> dict:
    """Backtest basis arb para un símbolo."""
    df = load_aligned(symbol)
    n = df.height
    if n < LOOKBACK + 10:
        return {"symbol": symbol, "error": "insufficient data"}

    fr = df["funding_rate"].to_numpy()
    spot = df["spot_close"].to_numpy()
    perp = df["perp_close"].to_numpy()

    entry_thr = rolling_quantile(fr, LOOKBACK, ENTRY_PCT)
    exit_thr = rolling_quantile(fr, LOOKBACK, EXIT_PCT)

    pnl_series = np.zeros(n)
    trade_pnls = []
    position = False
    entry_idx = -1
    hold_count = 0

    for i in range(n):
        if np.isnan(entry_thr[i]):
            continue
        if not position:
            # Entrar si funding > umbral entrada.
            if fr[i] > entry_thr[i] and fr[i] > 0:
                position = True
                entry_idx = i
                hold_count = 0
                # Costo de entrada: 2 fills (spot + perp).
                pnl_series[i] -= ROUND_TRIP_COST / 2
        else:
            # Cobrar funding cada intervalo (si funding > 0, short perp lo recibe).
            pnl_series[i] += fr[i]
            hold_count += 1
            # Salir si funding < mediana o max hold.
            should_exit = (fr[i] < exit_thr[i]) or (hold_count >= MAX_HOLD)
            if should_exit:
                # Costo de salida: 2 fills.
                pnl_series[i] -= ROUND_TRIP_COST / 2
                trade_pnl = pnl_series[i]
                # Sumar PnL de trade completo.
                trade_pnls.append(float(pnl_series[entry_idx : i + 1].sum()))
                position = False
                entry_idx = -1

    # Si sigue abierto al final, cerrar y registrar.
    if position and entry_idx >= 0:
        pnl_series[-1] -= ROUND_TRIP_COST / 2
        trade_pnls.append(float(pnl_series[entry_idx:].sum()))

    # PnL por intervalo de funding (3 por día).
    metrics = summary(pnl_series, np.array(trade_pnls), periods_per_year=3 * 365)
    metrics["symbol"] = symbol
    return metrics


def main() -> int:
    print(f"=== Backtest basis funding arb {START}→{END} ===")
    print(f"Params: lookback={LOOKBACK}  entry_p{ENTRY_PCT}  exit_p{EXIT_PCT}  "
          f"max_hold={MAX_HOLD}  round_trip_cost={ROUND_TRIP_COST*1e4:.0f}bps")
    print("-" * 80)
    rows = []
    for sym in UNIVERSE:
        try:
            m = backtest_symbol(sym)
            if "error" in m:
                print(f"{sym:10s}  ERROR: {m['error']}")
                continue
            rows.append(m)
            print(
                f"{sym:10s}  ret={m['total_return_pct']:+7.2f}%  "
                f"ann={m['ann_return_pct']:+6.2f}%  sharpe={m['sharpe']:+6.2f}  "
                f"sort={m['sortino']:+6.2f}  maxDD={m['max_drawdown_pct']:5.2f}%  "
                f"trades={m['n_trades']:4d}  hit={m['hit_rate_pct']:5.1f}%  "
                f"EV={m['ev_per_trade_bps']:+7.1f}bps  exp={m['exposure_pct']:5.1f}%"
            )
        except Exception as e:
            print(f"{sym:10s}  ERROR: {e}")

    if not rows:
        print("\nSin resultados.")
        return 1
    df = pl.DataFrame(rows)
    # Reordenar columnas.
    cols = ["symbol", "total_return_pct", "ann_return_pct", "sharpe", "sortino",
            "max_drawdown_pct", "n_trades", "hit_rate_pct", "ev_per_trade_bps", "exposure_pct"]
    df = df.select(cols).sort("sharpe", descending=True)
    df.write_csv(RESULTS_DIR / "backtest_per_symbol.csv")
    print("-" * 80)
    print(f"\nGuardado: {RESULTS_DIR}/backtest_per_symbol.csv")
    print(f"\n=== Aggregate (suma simple de PnL series asumiendo $1 por símbolo) ===")
    n_syms = len(rows)
    avg_ann = np.mean([r["ann_return_pct"] for r in rows])
    avg_sharpe = np.mean([r["sharpe"] for r in rows])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in rows])
    total_trades = sum(r["n_trades"] for r in rows)
    print(f"  símbolos: {n_syms}  avg_ann={avg_ann:+.2f}%  avg_sharpe={avg_sharpe:+.2f}  "
          f"avg_maxDD={avg_dd:.2f}%  total_trades={total_trades}")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
