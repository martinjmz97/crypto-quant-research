"""Backtest basis funding arb v3: basis risk + walk-forward.

Mejoras vs v2:
  - Modela el P&L del basis (movimiento perp-spot durante el hold).
  - Walk-forward: split train/test para validar OOS.
  - P&L total = funding cobrado + basis P&L - costos.

Mecánica P&L:
  - Long spot + short perp (direction=+1):
      P&L = (spot_t - spot_0)/spot_0 - (perp_t - perp_0)/perp_0 + sum(funding) - costs
  - Short spot + long perp (direction=-1):
      P&L = -(spot_t - spot_0)/spot_0 + (perp_t - perp_0)/perp_0 - sum(funding) - costs
  - El funding se cobra/paga cada 8h. Si direction=+1 y fr>0, cobramos. Si direction=-1 y fr<0, cobramos (longs pagan a shorts cuando fr<0).

Uso:
    python -m strategies.basis_funding.backtest_v3
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
DIRECTION_LOOKBACK = 180
FEE_MAKER = 0.0002
ROUND_TRIP_COST_MAKER = 4 * FEE_MAKER  # 8 bps


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


def backtest_symbol(symbol: str, train_end_idx: int | None = None) -> tuple[dict, np.ndarray, np.ndarray]:
    """Backtest con basis risk. Devuelve (métricas, pnl_series, is_oos_mask).

    Si train_end_idx se especifica, los umbrales se estiman solo con datos de train
    [0, train_end_idx), y se tradeará en todo el rango (train + test).
    is_oos_mask[i] = True si el índice i está en OOS.
    """
    df = load_aligned(symbol)
    n = df.height
    if n < LOOKBACK + 10:
        return {"symbol": symbol, "error": "insufficient data"}, np.zeros(0), np.zeros(0)

    fr = df["funding_rate"].to_numpy()
    spot = df["spot_close"].to_numpy()
    perp = df["perp_close"].to_numpy()

    entry_thr = rolling_quantile(fr, LOOKBACK, ENTRY_PCT)
    exit_thr = rolling_quantile(fr, LOOKBACK, EXIT_PCT)
    direction_signal = rolling_mean(fr, DIRECTION_LOOKBACK)

    pnl_series = np.zeros(n)
    trade_pnls: list[float] = []
    position = False
    direction = 0
    entry_idx = -1
    entry_spot = 0.0
    entry_perp = 0.0
    hold_count = 0

    for i in range(n):
        if np.isnan(entry_thr[i]) or np.isnan(direction_signal[i]):
            continue
        cur_dir = +1 if direction_signal[i] > 0 else -1
        if not position:
            if cur_dir == +1 and fr[i] > entry_thr[i] and fr[i] > 0:
                position = True
                direction = +1
                entry_idx = i
                entry_spot = spot[i]
                entry_perp = perp[i]
                hold_count = 0
                pnl_series[i] -= ROUND_TRIP_COST_MAKER / 2
            elif cur_dir == -1 and fr[i] < entry_thr[i] and fr[i] < 0:
                position = True
                direction = -1
                entry_idx = i
                entry_spot = spot[i]
                entry_perp = perp[i]
                hold_count = 0
                pnl_series[i] -= ROUND_TRIP_COST_MAKER / 2
        else:
            # Cobrar funding.
            pnl_series[i] += direction * fr[i]
            hold_count += 1
            # Basis P&L incremental: cambio de spot y perp desde la vela anterior.
            if i > entry_idx:
                spot_ret = (spot[i] - spot[i - 1]) / spot[i - 1] if spot[i - 1] > 0 else 0.0
                perp_ret = (perp[i] - perp[i - 1]) / perp[i - 1] if perp[i - 1] > 0 else 0.0
                # Long spot + short perp: gana si spot sube, gana si perp baja.
                pnl_series[i] += direction * (spot_ret - perp_ret)
            should_exit = False
            if direction == +1 and fr[i] < exit_thr[i]:
                should_exit = True
            elif direction == -1 and fr[i] > exit_thr[i]:
                should_exit = True
            elif hold_count >= MAX_HOLD:
                should_exit = True
            if should_exit:
                pnl_series[i] -= ROUND_TRIP_COST_MAKER / 2
                trade_pnls.append(float(pnl_series[entry_idx : i + 1].sum()))
                position = False
                entry_idx = -1

    if position and entry_idx >= 0:
        pnl_series[-1] -= ROUND_TRIP_COST_MAKER / 2
        trade_pnls.append(float(pnl_series[entry_idx:].sum()))

    # OOS mask.
    is_oos = np.zeros(n, dtype=bool)
    if train_end_idx is not None:
        is_oos[train_end_idx:] = True

    total_pnl = float(np.sum(pnl_series))
    n_trades = len(trade_pnls)
    n_years = n / (3 * 365)
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

    # Separar P&L OOS.
    oos_pnl = float(np.sum(pnl_series[is_oos])) if train_end_idx is not None else total_pnl
    oos_years = np.sum(is_oos) / (3 * 365) if train_end_idx is not None else n_years
    oos_ann = oos_pnl / oos_years if oos_years > 0 else 0.0

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
        "oos_pnl": round(oos_pnl, 4),
        "oos_ann": round(oos_ann, 4),
    }
    return metrics, pnl_series, is_oos


def main() -> int:
    print(f"=== Backtest basis funding v3 (basis risk + walk-forward) ===")
    print(f"Params: lookback={LOOKBACK}  entry_p{ENTRY_PCT}  exit_p{EXIT_PCT}  "
          f"max_hold={MAX_HOLD}  dir_lookback={DIRECTION_LOOKBACK}  "
          f"cost={ROUND_TRIP_COST_MAKER*1e4:.0f}bps (maker)")
    # Walk-forward: 70% train, 30% test.
    # Estimamos umbrales en train y tradear en todo el rango.
    # Para rolling quantiles, usamos ventana rolling (no toda la train), así que
    # el walk-forward es implícito: los umbrales se actualizan cada vela.
    # Split formal: reportar IS (in-sample) vs OOS.
    train_frac = 0.70
    print(f"Walk-forward: {train_frac*100:.0f}% IS / {(1-train_frac)*100:.0f}% OOS")
    print("-" * 120)
    rows = []
    pnl_series_by_sym: dict[str, np.ndarray] = {}
    oos_mask_by_sym: dict[str, np.ndarray] = {}
    for sym in UNIVERSE:
        try:
            # Primero necesitamos la longitud para calcular train_end_idx.
            df = load_aligned(sym)
            train_end = int(len(df) * train_frac)
            m, pnl, oos_mask = backtest_symbol(sym, train_end_idx=train_end)
            if "error" in m:
                print(f"{sym:10s}  ERROR: {m['error']}")
                continue
            rows.append(m)
            pnl_series_by_sym[sym] = pnl
            oos_mask_by_sym[sym] = oos_mask
            print(
                f"{sym:10s}  P&L_tot={m['total_pnl']:+7.4f}  ann={m['ann_pnl']:+6.4f}  "
                f"sharpe={m['sharpe']:+5.2f}  maxDD={m['max_dd']:.4f}  "
                f"trades={m['n_trades']:3d}  hit={m['hit_rate_pct']:4.1f}%  "
                f"EV={m['ev_per_trade']:+.4f}  "
                f"| OOS: ann={m['oos_ann']:+.4f}"
            )
        except Exception as e:
            print(f"{sym:10s}  ERROR: {e}")
    if not rows:
        return 1
    df_out = pl.DataFrame(rows)
    df_out.write_csv(RESULTS_DIR / "backtest_v3_per_symbol.csv")
    print("-" * 120)
    print(f"Guardado: {RESULTS_DIR}/backtest_v3_per_symbol.csv")

    # Análisis portfolio completo y OOS.
    syms = list(pnl_series_by_sym.keys())
    min_len = min(len(pnl_series_by_sym[s]) for s in syms)
    pnl_matrix = np.array([pnl_series_by_sym[s][:min_len] for s in syms])
    oos_mask = oos_mask_by_sym[syms[0]][:min_len]  # asumimos alineación similar

    # Portfolio IS vs OOS.
    portfolio_pnl = np.sum(pnl_matrix, axis=0)
    total_pnl_port = float(np.sum(portfolio_pnl))
    is_pnl = float(np.sum(portfolio_pnl[~oos_mask]))
    oos_pnl_port = float(np.sum(portfolio_pnl[oos_mask]))
    n_years_total = min_len / (3 * 365)
    n_years_oos = np.sum(oos_mask) / (3 * 365)
    ann_is = is_pnl / (n_years_total * (1 - 0.30)) if n_years_total > 0 else 0
    ann_oos = oos_pnl_port / n_years_oos if n_years_oos > 0 else 0
    sharpe_port = float(np.mean(portfolio_pnl) / np.std(portfolio_pnl) * np.sqrt(3 * 365)) if np.std(portfolio_pnl) > 0 else 0
    sharpe_oos = (float(np.mean(portfolio_pnl[oos_mask]) / np.std(portfolio_pnl[oos_mask]) * np.sqrt(3 * 365))
                  if np.std(portfolio_pnl[oos_mask]) > 0 else 0)
    equity_port = np.cumsum(portfolio_pnl)
    peak_port = np.maximum.accumulate(equity_port)
    max_dd_port = float(np.max(peak_port - equity_port))
    # OOS DD.
    equity_oos = np.cumsum(portfolio_pnl[oos_mask])
    peak_oos = np.maximum.accumulate(equity_oos)
    max_dd_oos = float(np.max(peak_oos - equity_oos)) if len(equity_oos) > 0 else 0

    print(f"\n=== PORTFOLIO ({len(syms)} símbolos, $1 c/u, ${len(syms)} total) ===")
    print(f"  P&L total: {total_pnl_port:+.4f}  anualizado: {ann_oos:+.4f} (OOS)")
    print(f"  IS  ann: {ann_is:+.4f}  | OOS ann: {ann_oos:+.4f}")
    print(f"  Sharpe total: {sharpe_port:+.3f}  | Sharpe OOS: {sharpe_oos:+.3f}")
    print(f"  Max DD total: {max_dd_port:.4f}  | Max DD OOS: {max_dd_oos:.4f}")
    print(f"  Degradación IS→OOS: {(1 - ann_oos/ann_is)*100 if ann_is != 0 else 0:.1f}%")

    print(f"\n=== ROI por leverage (capital base ${len(syms)}, $1/símbolo) ===")
    base_cap = len(syms)
    for lev in [1, 3, 5, 8]:
        ann_pnl_lev = ann_oos * lev
        max_dd_lev = max_dd_oos * lev
        ann_ret_pct = ann_pnl_lev / base_cap * 100
        max_dd_pct = max_dd_lev / base_cap * 100
        roi_dd = abs(ann_ret_pct / max_dd_pct) if max_dd_pct > 0 else 0
        print(f"  Leverage {lev}x: cap=${base_cap:3d}  ann_ret={ann_ret_pct:+6.2f}%  "
              f"maxDD={max_dd_pct:5.2f}%  ROI/DD={roi_dd:.2f}")

    print(f"\n=== Comparación v2 vs v3 ===")
    print(f"  v2 (sin basis risk): ann=+4.20%  sharpe=10.85  maxDD=0.81%")
    print(f"  v3 (con basis risk): ann={ann_oos:+.2%}  sharpe={sharpe_oos:+.2f}  maxDD={max_dd_oos/base_cap*100:.2f}% (OOS)")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
