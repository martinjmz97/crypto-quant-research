"""Backtest cluster mean reversion intradía.

Señal:
  - Beta rolling: ret_alt = alpha + beta * ret_BTC + eps (ventana 288 = 24h)
  - Residual acumulado: suma de eps en últimas 6 velas (30 min)
  - Z-score: (residual - mean_rolling) / std_rolling (ventana 288)

Reglas:
  - Entrada: z > +2 → short alt; z < -2 → long alt
  - Salida: z cruza 0 (cierre), o max hold 12 velas (1h)
  - Stop: |z| > 3.5 → cortar pérdida (sigue despegándose)
  - Costs: 20 bps round-trip (4 fills * 0.05% taker)
  - Notional $1 por trade, sin leverage

Uso:
    python -m strategies.cluster_reversion.backtest
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import _cache_path
from src.metrics import summary
from src.stats import rolling_zscore
from src.universe_lead_lag import LEAD_LAG_UNIVERSE, LEADER
from strategies.cluster_reversion.eda import load_returns, rolling_beta, rolling_sum

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TIMEFRAME = "5m"

BETA_WINDOW = 288
RESID_K = 6
Z_WINDOW = 288
ENTRY_Z = 2.0
EXIT_Z = 0.0
STOP_Z = 3.5
MAX_HOLD = 12
ROUND_TRIP_COST = 0.0020  # 20 bps


def compute_z(alt_ret: np.ndarray, btc_ret: np.ndarray) -> np.ndarray:
    beta, alpha = rolling_beta(alt_ret, btc_ret, BETA_WINDOW)
    eps = alt_ret - alpha - beta * btc_ret
    eps_cum = rolling_sum(eps, RESID_K)
    z = rolling_zscore(eps_cum, Z_WINDOW)
    return z


def backtest_symbol(symbol: str, btc_ret: np.ndarray, btc_df: pl.DataFrame) -> dict:
    alt_df = load_returns(symbol)
    joined = btc_df.drop("ret").join(
        alt_df, on="open_time", how="inner", suffix="_alt"
    ).sort("open_time")
    # Precios para P&L.
    alt_close = joined["close_alt"].to_numpy()
    # Retornos alineados.
    btc_r = np.log(joined["close"].to_numpy())
    alt_r = np.log(joined["close_alt"].to_numpy())
    btc_ret_a = np.diff(btc_r)
    alt_ret_a = np.diff(alt_r)
    n = len(alt_ret_a)
    z = compute_z(alt_ret_a, btc_ret_a)

    # PnL por vela: si posición abierta, PnL = -direction * alt_ret[t]
    # (short: gana si alt baja; long: gana si alt sube).
    pnl_series = np.zeros(n)
    trade_pnls: list[float] = []
    position = 0  # 0=flat, +1=long, -1=short
    entry_idx = -1
    entry_price = 0.0
    hold = 0

    for i in range(n):
        if np.isnan(z[i]):
            continue
        if position == 0:
            if z[i] > ENTRY_Z:
                position = -1
                entry_idx = i
                entry_price = alt_close[i + 1]  # ejecución siguiente vela
                hold = 0
                pnl_series[i] -= ROUND_TRIP_COST / 2
            elif z[i] < -ENTRY_Z:
                position = +1
                entry_idx = i
                entry_price = alt_close[i + 1]
                hold = 0
                pnl_series[i] -= ROUND_TRIP_COST / 2
        else:
            hold += 1
            # PnL realizado parcial: cambio de precio * dirección.
            cur_price = alt_close[i + 1] if i + 1 < len(alt_close) else alt_close[i]
            ret_trade = (cur_price - entry_price) / entry_price if entry_price > 0 else 0.0
            # PnL incremental = dirección * retorno desde vela anterior.
            if i + 1 < len(alt_close):
                prev_price = alt_close[i]
                pnl_inc = position * (cur_price - prev_price) / prev_price if prev_price > 0 else 0.0
                pnl_series[i] += pnl_inc
            # Salida?
            should_exit = False
            if abs(z[i]) < abs(EXIT_Z) + 0.001:
                should_exit = True
            elif abs(z[i]) > STOP_Z:
                should_exit = True
            elif hold >= MAX_HOLD:
                should_exit = True
            if should_exit:
                pnl_series[i] -= ROUND_TRIP_COST / 2
                trade_pnl = float(pnl_series[entry_idx : i + 1].sum())
                trade_pnls.append(trade_pnl)
                position = 0
                entry_idx = -1

    if position != 0 and entry_idx >= 0:
        pnl_series[-1] -= ROUND_TRIP_COST / 2
        trade_pnls.append(float(pnl_series[entry_idx:].sum()))

    # Métricas. Período = 5m -> 365*24*12 = 105120 por año.
    metrics = summary(pnl_series, np.array(trade_pnls), periods_per_year=365 * 24 * 12)
    metrics["symbol"] = symbol
    return metrics


def main() -> int:
    print(f"=== Backtest cluster reversion, timeframe {TIMEFRAME} ===")
    print(f"Params: beta_w={BETA_WINDOW} resid_k={RESID_K} z_w={Z_WINDOW} "
          f"entry={ENTRY_Z} exit={EXIT_Z} stop={STOP_Z} max_hold={MAX_HOLD} "
          f"cost={ROUND_TRIP_COST*1e4:.0f}bps")
    print("-" * 90)
    btc_df = load_returns(LEADER)
    btc_ret = btc_df["ret"].to_numpy()
    rows = []
    for alt in LEAD_LAG_UNIVERSE:
        try:
            m = backtest_symbol(alt, btc_ret, btc_df)
            rows.append(m)
            print(
                f"{alt:10s}  ret={m['total_return_pct']:+8.2f}%  "
                f"ann={m['ann_return_pct']:+7.2f}%  sharpe={m['sharpe']:+6.2f}  "
                f"sort={m['sortino']:+6.2f}  maxDD={m['max_drawdown_pct']:6.2f}%  "
                f"trades={m['n_trades']:5d}  hit={m['hit_rate_pct']:5.1f}%  "
                f"EV={m['ev_per_trade_bps']:+7.1f}bps  exp={m['exposure_pct']:5.1f}%"
            )
        except Exception as e:
            print(f"{alt:10s}  ERROR: {e}")
    if not rows:
        print("\nSin resultados.")
        return 1
    df = pl.DataFrame(rows)
    cols = ["symbol", "total_return_pct", "ann_return_pct", "sharpe", "sortino",
            "max_drawdown_pct", "n_trades", "hit_rate_pct", "ev_per_trade_bps", "exposure_pct"]
    df = df.select(cols).sort("ann_return_pct", descending=True)
    df.write_csv(RESULTS_DIR / "backtest_per_symbol.csv")
    print("-" * 90)
    print(f"\nGuardado: {RESULTS_DIR}/backtest_per_symbol.csv")
    # Aggregate.
    avg_ann = np.mean([r["ann_return_pct"] for r in rows])
    avg_sharpe = np.mean([r["sharpe"] for r in rows])
    avg_dd = np.mean([r["max_drawdown_pct"] for r in rows])
    avg_hit = np.mean([r["hit_rate_pct"] for r in rows])
    total_trades = sum(r["n_trades"] for r in rows)
    avg_ev = np.mean([r["ev_per_trade_bps"] for r in rows])
    print(f"\n=== Aggregate ({len(rows)} alts) ===")
    print(f"  avg_ann={avg_ann:+.2f}%  avg_sharpe={avg_sharpe:+.2f}  avg_maxDD={avg_dd:.2f}%  "
          f"avg_hit={avg_hit:.1f}%  total_trades={total_trades}  avg_EV={avg_ev:+.1f}bps")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
