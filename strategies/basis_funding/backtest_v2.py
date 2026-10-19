"""Backtest basis funding arb v2: maker orders + dirección adaptativa.

Mejoras vs v1:
  - Maker orders: 4 * 0.02% = 8 bps round-trip (vs 20 bps taker).
  - Dirección adaptativa: si funding rolling mean < 0 → invertir (short spot + long perp).
  - Devuelve pnl_series por símbolo para análisis de correlación y diversificación.

Uso:
    python -m strategies.basis_funding.backtest_v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import _cache_path, load_funding
from src.universe import UNIVERSE

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-01-01"
END = "2024-12-31"
LOOKBACK = 90
ENTRY_PCT = 0.80
EXIT_PCT = 0.50
MAX_HOLD = 60
DIRECTION_LOOKBACK = 180  # ventana para decidir dirección (funding rolling mean)


def load_aligned(symbol: str) -> pl.DataFrame:
    fund = load_funding(symbol, START, END).select("funding_time", "funding_rate", "mark_price")
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
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) > 0:
            out[i] = np.quantile(valid, q)
    return out


def rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        w = arr[i - window + 1 : i + 1]
        valid = w[~np.isnan(w)]
        if len(valid) > 0:
            out[i] = np.mean(valid)
    return out


def backtest_symbol(symbol: str, fee_taker: float = 0.0005, fee_maker: float = 0.0002,
                     use_maker: bool = True) -> tuple[dict, np.ndarray]:
    """Backtest con dirección adaptativa. Devuelve (métricas, pnl_series)."""
    df = load_aligned(symbol)
    n = df.height
    if n < LOOKBACK + 10:
        return {"symbol": symbol, "error": "insufficient data"}, np.zeros(0)

    fr = df["funding_rate"].to_numpy()
    round_trip_cost = 4 * (fee_maker if use_maker else fee_taker)

    entry_thr = rolling_quantile(fr, LOOKBACK, ENTRY_PCT)
    exit_thr = rolling_quantile(fr, LOOKBACK, EXIT_PCT)
    direction_signal = rolling_mean(fr, DIRECTION_LOOKBACK)  # >0 long spot/short perp; <0 invert

    pnl_series = np.zeros(n)
    trade_pnls: list[float] = []
    position = False
    direction = 0  # +1 = long spot/short perp, -1 = short spot/long perp
    entry_idx = -1
    hold_count = 0

    for i in range(n):
        if np.isnan(entry_thr[i]) or np.isnan(direction_signal[i]):
            continue
        # Dirección adaptativa.
        cur_dir = +1 if direction_signal[i] > 0 else -1
        if not position:
            # Entrar si funding cruza umbral en la dirección correcta.
            if cur_dir == +1 and fr[i] > entry_thr[i] and fr[i] > 0:
                position = True
                direction = +1
                entry_idx = i
                hold_count = 0
                pnl_series[i] -= round_trip_cost / 2
            elif cur_dir == -1 and fr[i] < entry_thr[i] and fr[i] < 0:
                # Dirección invertida: short spot + long perp, cobrar funding negativo.
                position = True
                direction = -1
                entry_idx = i
                hold_count = 0
                pnl_series[i] -= round_trip_cost / 2
        else:
            # Cobrar funding (direction * funding: si short perp y funding>0, cobrás).
            pnl_series[i] += direction * fr[i]
            hold_count += 1
            # Salida: funding cruza mediana (en la dirección contraria) o max hold.
            should_exit = False
            if direction == +1 and fr[i] < exit_thr[i]:
                should_exit = True
            elif direction == -1 and fr[i] > exit_thr[i]:
                should_exit = True
            elif hold_count >= MAX_HOLD:
                should_exit = True
            if should_exit:
                pnl_series[i] -= round_trip_cost / 2
                trade_pnls.append(float(pnl_series[entry_idx : i + 1].sum()))
                position = False
                entry_idx = -1

    if position and entry_idx >= 0:
        pnl_series[-1] -= round_trip_cost / 2
        trade_pnls.append(float(pnl_series[entry_idx:].sum()))

    total_pnl = float(np.sum(pnl_series))
    n_trades = len(trade_pnls)
    n_intervals = n
    n_years = n_intervals / (3 * 365)  # 3 funding intervals por día
    ann_pnl = total_pnl / n_years if n_years > 0 else 0.0
    if len(pnl_series) > 1 and np.std(pnl_series) > 0:
        sharpe = float(np.mean(pnl_series) / np.std(pnl_series) * np.sqrt(3 * 365))
    else:
        sharpe = 0.0
    equity = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(equity)
    max_dd = float(np.max(peak - equity)) if len(equity) > 0 else 0.0
    hit = float(np.mean(np.array(trade_pnls) > 0)) * 100 if n_trades > 0 else 0.0
    ev = float(np.mean(trade_pnls)) if n_trades > 0 else 0.0
    exposure = float(np.mean(pnl_series != 0)) * 100

    metrics = {
        "symbol": symbol,
        "total_pnl": round(total_pnl, 4),
        "ann_pnl": round(ann_pnl, 4),
        "sharpe": round(sharpe, 3),
        "max_dd": round(max_dd, 4),
        "n_trades": n_trades,
        "hit_rate_pct": round(hit, 2),
        "ev_per_trade": round(ev, 4),
        "exposure_pct": round(exposure, 2),
    }
    return metrics, pnl_series


def main() -> int:
    fee_mode = "maker"
    print(f"=== Backtest basis funding v2 ({fee_mode}, dirección adaptativa) ===")
    print(f"Params: lookback={LOOKBACK}  entry_p{ENTRY_PCT}  exit_p{EXIT_PCT}  "
          f"max_hold={MAX_HOLD}  dir_lookback={DIRECTION_LOOKBACK}")
    print("-" * 110)
    rows = []
    pnl_series_by_sym: dict[str, np.ndarray] = {}
    for sym in UNIVERSE:
        try:
            m, pnl = backtest_symbol(sym, use_maker=True)
            if "error" in m:
                print(f"{sym:10s}  ERROR: {m['error']}")
                continue
            rows.append(m)
            pnl_series_by_sym[sym] = pnl
            print(
                f"{sym:10s}  P&L_tot={m['total_pnl']:+7.4f}  ann={m['ann_pnl']:+6.4f}  "
                f"sharpe={m['sharpe']:+5.2f}  maxDD={m['max_dd']:.4f}  "
                f"trades={m['n_trades']:3d}  hit={m['hit_rate_pct']:4.1f}%  "
                f"EV={m['ev_per_trade']:+.4f}  exp={m['exposure_pct']:4.1f}%"
            )
        except Exception as e:
            print(f"{sym:10s}  ERROR: {e}")
    if not rows:
        return 1
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_DIR / "backtest_v2_per_symbol.csv")
    print("-" * 110)
    print(f"Guardado: {RESULTS_DIR}/backtest_v2_per_symbol.csv")

    # Análisis de diversificación.
    print(f"\n=== Análisis de diversificación ({len(rows)} símbolos) ===")
    # Alinear PnL series por índice (todas mismas longitudes).
    syms = list(pnl_series_by_sym.keys())
    min_len = min(len(pnl_series_by_sym[s]) for s in syms)
    pnl_matrix = np.array([pnl_series_by_sym[s][:min_len] for s in syms])
    # Correlación promedio entre pares.
    corr = np.corrcoef(pnl_matrix)
    # Promedio de off-diagonal.
    n_sym = len(syms)
    off_diag = corr[~np.eye(n_sym, dtype=bool)]
    avg_corr = float(np.mean(off_diag))
    print(f"  Correlación promedio entre PnL series: {avg_corr:+.3f}")
    print(f"  Matrix (primeras 5):")
    for i in range(min(5, n_sym)):
        row_str = " ".join(f"{corr[i,j]:+.2f}" for j in range(min(5, n_sym)))
        print(f"    {syms[i]:10s} {row_str}")

    # Portfolio aggregate: suma de PnL (asumiendo $1 por símbolo).
    portfolio_pnl = np.sum(pnl_matrix, axis=0)
    total_pnl_port = float(np.sum(portfolio_pnl))
    n_years = min_len / (3 * 365)
    ann_pnl_port = total_pnl_port / n_years
    sharpe_port = float(np.mean(portfolio_pnl) / np.std(portfolio_pnl) * np.sqrt(3 * 365)) if np.std(portfolio_pnl) > 0 else 0
    equity_port = np.cumsum(portfolio_pnl)
    peak_port = np.maximum.accumulate(equity_port)
    max_dd_port = float(np.max(peak_port - equity_port))
    print(f"\n  PORTFOLIO ($1 por símbolo, {n_sym} símbolos, ${n_sym} total):")
    print(f"    P&L total: {total_pnl_port:+.4f}  anualizado: {ann_pnl_port:+.4f}")
    print(f"    Sharpe portfolio: {sharpe_port:+.3f}")
    print(f"    Max DD portfolio: {max_dd_port:.4f}")
    # Comparar con suma de DDs individuales.
    sum_dd = sum(m["max_dd"] for m in rows)
    print(f"    Suma DDs individuales: {sum_dd:.4f}  (sin diversificación)")
    print(f"    DD portfolio real:    {max_dd_port:.4f}  (con diversificación)")
    print(f"    Beneficio diversificación: {(1 - max_dd_port/sum_dd)*100:.1f}% menos DD")

    # ROI con capital.
    print(f"\n=== ROI por nivel de leverage (capital base ${n_sym}, $1/símbolo) ===")
    # Capital fijo = $n_sym. Leverage multiplica el notional, no el capital.
    base_cap = n_sym  # $1 por símbolo, sin leverage
    for lev in [1, 3, 5, 8]:
        ann_pnl_lev = ann_pnl_port * lev
        max_dd_lev = max_dd_port * lev
        ann_ret_pct = ann_pnl_lev / base_cap * 100
        max_dd_pct = max_dd_lev / base_cap * 100
        roi_dd = abs(ann_ret_pct / max_dd_pct) if max_dd_pct > 0 else 0
        print(f"  Leverage {lev}x: cap=${base_cap:3d}  ann_ret={ann_ret_pct:+6.2f}%  "
              f"maxDD={max_dd_pct:5.2f}%  ROI/DD={roi_dd:.2f}")

    # Comparación v1 vs v2.
    print(f"\n=== Comparación v1 (taker, sin dir adaptativa) vs v2 (maker, con dir adaptativa) ===")
    print(f"  v1: avg_ann=2.57%  avg_sharpe=3.82  avg_maxDD=2.12%  (por símbolo individual)")
    print(f"  v2 portfolio: ann={ann_pnl_port/base_cap*100:+.2f}%  sharpe={sharpe_port:+.2f}  "
          f"maxDD={max_dd_port/base_cap*100:.2f}%  (portfolio diversificado)")
    print(f"  Mejora retorno: {(ann_pnl_port/base_cap*100 - 2.57):+.2f} pp")
    print(f"  Mejora Sharpe: {sharpe_port - 3.82:+.2f}")
    print(f"  Mejora DD: {(2.12 - max_dd_port/base_cap*100):+.2f} pp (menor DD)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
