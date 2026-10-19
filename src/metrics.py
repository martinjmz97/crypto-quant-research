"""Métricas de performance para backtests."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 365) -> float:
    """Sharpe anualizado. returns: array de retornos por período."""
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0.0:
        return 0.0
    return float(mu / sigma * np.sqrt(periods_per_year))


@njit(cache=True, fastmath=True)
def sortino_ratio(returns: np.ndarray, periods_per_year: int = 365) -> float:
    """Sortino anualizado (solo downside deviation)."""
    if len(returns) < 2:
        return 0.0
    mu = returns.mean()
    downside = np.where(returns < 0, returns, 0.0)
    dd = np.sqrt((downside * downside).mean())
    if dd == 0.0:
        return 0.0
    return float(mu / dd * np.sqrt(periods_per_year))


@njit(cache=True, fastmath=True)
def max_drawdown(equity: np.ndarray) -> float:
    """Máximo drawdown sobre curva de equity."""
    if len(equity) == 0:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for i in range(len(equity)):
        if equity[i] > peak:
            peak = equity[i]
        dd = (peak - equity[i]) / peak if peak > 0.0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


@njit(cache=True, fastmath=True)
def hit_rate(pnl_per_trade: np.ndarray) -> float:
    """Porcentaje de trades ganadores."""
    if len(pnl_per_trade) == 0:
        return 0.0
    return float(np.mean(pnl_per_trade > 0.0))


def summary(pnl_series: np.ndarray, pnl_per_trade: np.ndarray,
            periods_per_year: int = 365, initial_capital: float = 1.0) -> dict:
    """Resumen completo de métricas.

    pnl_series: array de P&L por período (en dólares o fracción de capital).
    pnl_per_trade: P&L por trade cerrado (mismo unidad que pnl_series).
    initial_capital: solo usado para max_drawdown y return si pnl_series es $.
    """
    if len(pnl_series) == 0:
        return {}
    equity = initial_capital + np.cumsum(pnl_series)
    total_return_pct = float(equity[-1] / initial_capital - 1.0) * 100.0
    ann_return_pct = total_return_pct * (periods_per_year / len(pnl_series)) if len(pnl_series) > 0 else 0.0
    # Sharpe y Sortino usan los P&L directamente (no importa si son $ o fracción).
    n_years = len(pnl_series) / periods_per_year if periods_per_year > 0 else 1.0
    mu = float(np.mean(pnl_series))
    sigma = float(np.std(pnl_series))
    sharpe = (mu / sigma * np.sqrt(periods_per_year)) if sigma > 0 else 0.0
    downside = np.where(pnl_series < 0, pnl_series, 0.0)
    dd = np.sqrt((downside * downside).mean()) if len(downside) > 0 else 0.0
    sortino = (mu / dd * np.sqrt(periods_per_year)) if dd > 0 else 0.0
    mdd = max_drawdown(equity) * 100.0
    return {
        "total_return_pct": round(total_return_pct, 3),
        "ann_return_pct": round(ann_return_pct, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd, 3),
        "n_trades": len(pnl_per_trade),
        "hit_rate_pct": round(hit_rate(pnl_per_trade) * 100, 2) if len(pnl_per_trade) > 0 else 0.0,
        "ev_per_trade": round(float(np.mean(pnl_per_trade)) if len(pnl_per_trade) > 0 else 0.0, 3),
        "exposure_pct": round(float(np.mean(pnl_series != 0.0)) * 100, 2),
    }
