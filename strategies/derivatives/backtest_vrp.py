"""Backtest VRP: short straddle ATM proxy.

Mecánica (proxy simplificado, sin BS pricing):
  - Cada día t: si VRP_t > umbral_entry y no hay posición → short straddle ATM
    notional = $1 de vega expuesto (controlamos size después).
  - Hold N días (mantenemos el straddle).
  - P&L del straddle corto ≈ vega_notional * (RV_realizada_en_hold - IV_al_entrada) / 100
    (en puntos de vol, *1000 porque vega es $ per 1% vol, notional $1 vega = $1000)
  - Salida: después de hold_days días, o VRP < umbral_exit.
  - Costos: ~10 bps del notional por leg (premium de opción), 4 legs round-trip.

Proxy simplificado: el P&L lo aproximamos como la diferencia entre
la IV al entrada y la RV efectivamente realizada en los siguientes
hold_days días. Esto captura el core edge: si IV > RV_forward, el short vol gana.

Nota: esto NO modela gamma/vega/delta hedging. Es la aproximación más simple.
Para producción: implementar BS pricing + Greeks (futuro).

Uso:
    python -m strategies.derivatives.backtest_vrp
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
from numba import njit

from src.deribit_loader import load_dvol
from strategies.derivatives.rv_calculator import compute_rv, CACHE_DIR
from src.metrics import summary

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-10-09"
END = "2026-07-04"

# Parámetros.
ENTRY_VRP = 5.0    # entrar si VRP > 5 puntos de vol (pp)
EXIT_VRP = 0.0     # salir si VRP < 0
HOLD_DAYS = 7      # hold del straddle (días)
VEGA_NOTIONAL_USD = 1000.0  # $1000 por 1% vol move (=$1 de vega real)
# Costos por trade: 4 legs * 0.02% (maker) * 0.5 BTC notional ~= pequeño.
# Approximamos como costo fijo: 15 bps del vega_notional por round-trip.
ROUND_TRIP_COST_BPS = 15.0


def load_vrp_daily(currency: str, underlying: str) -> pl.DataFrame:
    """DataFrame diario con dvol, rv_close, vrp, y rv_forward (RV de los HOLD_DAYS siguientes)."""
    dvol = load_dvol(currency, START, END, resolution="86400")
    dvol = dvol.select(pl.col("ts").dt.date().alias("date"), pl.col("close").alias("dvol"))
    rv = compute_rv(underlying, window_hours=720)
    df = dvol.join(rv, on="date", how="inner").sort("date")
    df = df.with_columns((pl.col("dvol") - pl.col("rv_close")).alias("vrp"))
    return df


@njit(cache=True)
def rv_forward_from_hourly(log_ret_1h: np.ndarray, daily_indices: np.ndarray,
                             hold_hours: int) -> np.ndarray:
    """RV forward: vol anualizada de los siguientes hold_hours desde cada día.

    daily_indices[i] = índice en log_ret_1h del primer 1h del día i.
    Devuelve RV forward en % anualizada, NaN si no hay suficientes datos.
    """
    n_days = len(daily_indices)
    out = np.full(n_days, np.nan)
    pp_year = 365 * 24
    for i in range(n_days):
        start_idx = daily_indices[i]
        end_idx = start_idx + hold_hours
        if end_idx >= len(log_ret_1h):
            break
        window = log_ret_1h[start_idx:end_idx]
        valid = 0
        s = 0.0
        ss = 0.0
        for v in window:
            if not np.isnan(v):
                s += v
                ss += v * v
                valid += 1
        if valid < 2:
            continue
        m = s / valid
        var = ss / valid - m * m
        if var < 0:
            var = 0.0
        out[i] = np.sqrt(var) * np.sqrt(pp_year) * 100.0
    return out


def compute_rv_forward(symbol: str, hold_days: int, daily_dates: list) -> np.ndarray:
    """RV forward para cada día en daily_dates."""
    from src.data_loader import load_klines
    df = load_klines(symbol, "1h", START, END, market="perp").sort("open_time")
    df = df.with_columns(pl.col("close").log().diff().alias("log_ret"))
    log_ret = df["log_ret"].to_numpy()
    dates = df["open_time"].dt.date().to_numpy()
    # Índice del primer 1h de cada día en daily_dates.
    daily_indices = []
    j = 0
    for d in daily_dates:
        while j < len(dates) and dates[j] < d:
            j += 1
        if j < len(dates) and dates[j] == d:
            daily_indices.append(j)
        else:
            daily_indices.append(-1)
    daily_indices_np = np.array(daily_indices, dtype=np.int64)
    valid_mask = daily_indices_np >= 0
    out = np.full(len(daily_dates), np.nan)
    if valid_mask.sum() == 0:
        return out
    # Llamar a numba con índices válidos.
    valid_idx = daily_indices_np[valid_mask]
    rv_fwd = rv_forward_from_hourly(log_ret, valid_idx, hold_days * 24)
    out[valid_mask] = rv_fwd
    return out


def backtest_vrp(currency: str, underlying: str) -> dict:
    """Backtest VRP proxy. Devuelve métricas en dólares absolutos."""
    df = load_vrp_daily(currency, underlying)
    daily_dates = df["date"].to_list()
    rv_fwd = compute_rv_forward(underlying, HOLD_DAYS, daily_dates)
    df = df.with_columns(pl.Series(name="rv_forward", values=rv_fwd))

    n = df.height
    dvol = df["dvol"].to_numpy()
    vrp = df["vrp"].to_numpy()
    rv_fwd_arr = df["rv_forward"].to_numpy()

    pnl_series = np.zeros(n)
    trade_pnls: list[float] = []
    position = False
    entry_idx = -1
    entry_iv = 0.0
    hold_count = 0

    for i in range(n):
        if np.isnan(vrp[i]) or np.isnan(rv_fwd_arr[i]):
            continue
        if not position:
            if vrp[i] > ENTRY_VRP:
                position = True
                entry_idx = i
                entry_iv = dvol[i]
                hold_count = 0
                pnl_series[i] -= (ROUND_TRIP_COST_BPS / 2) * VEGA_NOTIONAL_USD / 10000.0
        else:
            hold_count += 1
            should_exit = (vrp[i] < EXIT_VRP) or (hold_count >= HOLD_DAYS)
            if should_exit:
                rv_realized = rv_fwd_arr[entry_idx]
                # Short vol P&L: gana si RV_forward < IV_entry. Por 1 punto de vol = $1 por $100 vega.
                trade_pnl = VEGA_NOTIONAL_USD * (entry_iv - rv_realized) / 100.0
                trade_pnl -= (ROUND_TRIP_COST_BPS / 2) * VEGA_NOTIONAL_USD / 10000.0
                pnl_series[i] += trade_pnl
                trade_pnls.append(float(trade_pnl))
                position = False
                entry_idx = -1

    if position and entry_idx >= 0:
        rv_realized = rv_fwd_arr[entry_idx] if not np.isnan(rv_fwd_arr[entry_idx]) else 0.0
        trade_pnl = VEGA_NOTIONAL_USD * (entry_iv - rv_realized) / 100.0
        trade_pnl -= (ROUND_TRIP_COST_BPS / 2) * VEGA_NOTIONAL_USD / 10000.0
        pnl_series[-1] += trade_pnl
        trade_pnls.append(float(trade_pnl))

    trade_pnls_arr = np.array(trade_pnls) if trade_pnls else np.zeros(0)
    # Métricas directas en dólares.
    total_pnl = float(np.sum(pnl_series))
    n_trades = len(trade_pnls)
    n_years = n / 365.0
    ann_pnl = total_pnl / n_years if n_years > 0 else 0.0
    # Sharpe diario: mean/std * sqrt(365).
    active_pnl = pnl_series[pnl_series != 0]  # días con trades cerrados
    if len(active_pnl) > 1:
        sharpe = float(np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(365)) if np.std(pnl_series) > 0 else 0.0
    else:
        sharpe = 0.0
    # Equity curve y max DD en dólares.
    equity = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(equity)
    dd_arr = (peak - equity)
    max_dd = float(np.max(dd_arr)) if len(dd_arr) > 0 else 0.0
    hit = float(np.mean(trade_pnls_arr > 0)) * 100 if n_trades > 0 else 0.0
    ev_trade = float(np.mean(trade_pnls_arr)) if n_trades > 0 else 0.0
    exposure = float(np.mean(pnl_series != 0)) * 100

    return {
        "currency": currency,
        "total_pnl_usd": round(total_pnl, 2),
        "ann_pnl_usd": round(ann_pnl, 2),
        "sharpe": round(sharpe, 3),
        "max_dd_usd": round(max_dd, 2),
        "n_trades": n_trades,
        "hit_rate_pct": round(hit, 2),
        "ev_per_trade_usd": round(ev_trade, 2),
        "exposure_pct": round(exposure, 2),
    }


def main() -> int:
    print(f"=== Backtest VRP proxy (short straddle ATM) ===")
    print(f"Params: entry_VRP>{ENTRY_VRP}pp  exit_VRP<{EXIT_VRP}  hold={HOLD_DAYS}d  "
          f"vega_notional=${VEGA_NOTIONAL_USD:.0f}  round_trip_cost={ROUND_TRIP_COST_BPS:.0f}bps")
    print("-" * 100)
    rows = []
    for currency, underlying in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        try:
            m = backtest_vrp(currency, underlying)
            rows.append(m)
            print(
                f"{currency:5s}  P&L_tot=${m['total_pnl_usd']:+9.0f}  ann=${m['ann_pnl_usd']:+8.0f}  "
                f"sharpe={m['sharpe']:+5.2f}  maxDD=${m['max_dd_usd']:+8.0f}  "
                f"trades={m['n_trades']:3d}  hit={m['hit_rate_pct']:4.1f}%  "
                f"EV=${m['ev_per_trade_usd']:+6.1f}  exp={m['exposure_pct']:4.1f}%"
            )
        except Exception as e:
            print(f"{currency:5s}  ERROR: {e}")
            import traceback; traceback.print_exc()
    if not rows:
        return 1
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_DIR / "backtest_vrp.csv")
    print(f"\nGuardado: {RESULTS_DIR}/backtest_vrp.csv")

    # Capital estimation.
    print(f"\n=== Estimación de capital inicial (vega_notional ${VEGA_NOTIONAL_USD}) ===")
    for r in rows:
        cur = r["currency"]
        n_trades = r["n_trades"]
        years = 2.74
        trades_per_year = n_trades / years
        ann_pnl = r["ann_pnl_usd"]
        # Margen Deribit short straddle ATM: ~$10k por 1 BTC straddle (proxy).
        margin_per_straddle = 10000.0
        print(f"\n  {cur}: {trades_per_year:.0f} trades/año")
        print(f"    P&L anual 1-straddle (vega ${VEGA_NOTIONAL_USD}): ${ann_pnl:+.0f}")
        for n_straddles in [1, 5, 10]:
            cap = n_straddles * margin_per_straddle
            ann_pnl_n = n_straddles * ann_pnl
            roi = ann_pnl_n / cap * 100
            print(f"    {n_straddles:2d} straddle(s): cap=${cap:6.0f}  ann_PnL=${ann_pnl_n:+7.0f}  ROI={roi:+5.1f}%")
    print()
    print("NOTAS:")
    print("  - El proxy NO modela gamma/vega/delta hedging. El P&L real será menor.")
    print("  - Margen conservador; con portfolio margin cap sería ~40% menor.")
    print("  - maxDD asume 1 straddle fijo; con N simultáneos escala linealmente.")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
