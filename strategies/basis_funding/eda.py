"""EDA del funding rate: distribución, persistencia, régimen, base premium.

Construye una tabla resumen por símbolo y guarda en results/eda_funding.csv.
También imprime estadísticas clave para decidir la regla de entrada.

Uso:
    python -m strategies.basis_funding.eda
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import _cache_path, load_funding, load_klines
from src.universe import UNIVERSE

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
START = "2023-01-01"
END = "2024-12-31"


def load_perp_close(symbol: str) -> pl.DataFrame:
    cache = _cache_path(symbol, "1h", "perp")
    return pl.read_parquet(cache).select("open_time", pl.col("close").alias("perp_close"))


def load_spot_close(symbol: str) -> pl.DataFrame:
    cache = _cache_path(symbol, "1h", "spot")
    return pl.read_parquet(cache).select("open_time", pl.col("close").alias("spot_close"))


def base_premium(symbol: str) -> pl.DataFrame:
    """Calcula base premium = (perp - spot) / spot en cada funding time."""
    fund = load_funding(symbol, START, END).select("funding_time", "funding_rate", "mark_price")
    perp = load_perp_close(symbol)
    spot = load_spot_close(symbol)
    # Para cada funding_time (00:00, 08:00, 16:00 UTC), tomar la vela 1h cerrada en ese instante.
    df = (
        fund.join(perp, left_on="funding_time", right_on="open_time", how="inner")
        .join(spot, left_on="funding_time", right_on="open_time", how="inner")
    )
    df = df.with_columns(
        ((pl.col("perp_close") - pl.col("spot_close")) / pl.col("spot_close")).alias("base_premium"),
    )
    return df


def symbol_summary(symbol: str) -> dict:
    df = base_premium(symbol)
    fr = df["funding_rate"]
    bp = df["base_premium"]
    # Funding acumulado anualizado: 3 funding/day * 365 = 1095 intervalos.
    n_intervals_year = 3 * 365
    annual_funding = fr.mean() * n_intervals_year
    # Sharpe crudo del funding (sin costos): mean/std * sqrt(intervals_per_year).
    sharpe_funding = (fr.mean() / fr.std()) * np.sqrt(n_intervals_year) if fr.std() > 0 else 0.0
    # Autocorrelación lag-1 del funding (persistencia).
    fr_arr = fr.to_numpy()
    fr_lag = fr_arr[:-1]
    fr_cur = fr_arr[1:]
    ac1 = float(np.corrcoef(fr_lag, fr_cur)[0, 1]) if len(fr_lag) > 2 else float("nan")
    # % de funding positivo (región donde longs pagan shorts).
    pct_pos = float((fr > 0).mean())
    return {
        "symbol": symbol,
        "n": df.height,
        "fund_mean_bps": round(fr.mean() * 1e4, 2),
        "fund_std_bps": round(fr.std() * 1e4, 2),
        "fund_annual_pct": round(annual_funding * 100, 2),
        "fund_sharpe": round(sharpe_funding, 3),
        "fund_ac1": round(ac1, 3),
        "pct_positive": round(pct_pos, 3),
        "base_premium_mean_bps": round(bp.mean() * 1e4, 2),
        "base_premium_std_bps": round(bp.std() * 1e4, 2),
    }


def main() -> int:
    print(f"=== EDA funding rate {START}→{END} ===\n")
    rows = []
    for sym in UNIVERSE:
        try:
            s = symbol_summary(sym)
            rows.append(s)
            print(
                f"{sym:10s}  fundμ={s['fund_mean_bps']:+7.2f}bps  "
                f"σ={s['fund_std_bps']:6.2f}  annual={s['fund_annual_pct']:+6.2f}%  "
                f"sharpe={s['fund_sharpe']:+5.2f}  ac1={s['fund_ac1']:.3f}  "
                f"%pos={s['pct_positive']:.2f}  baseμ={s['base_premium_mean_bps']:+7.2f}bps"
            )
        except Exception as e:
            print(f"{sym:10s}  ERROR: {e}")
    df = pl.DataFrame(rows)
    df.write_csv(RESULTS_DIR / "eda_funding.csv")
    print(f"\nGuardado: {RESULTS_DIR}/eda_funding.csv")

    # Top candidatos por Sharpe de funding.
    print("\n=== Top 10 por Sharpe de funding (más predecible) ===")
    top = df.sort("fund_sharpe", descending=True).head(10)
    with pl.Config(tbl_width_chars=160, float_precision=3):
        print(top)
    print("\n=== Top 10 por funding anual acumulado (más rentable bruto) ===")
    top_annual = df.sort("fund_annual_pct", descending=True).head(10)
    with pl.Config(tbl_width_chars=160, float_precision=3):
        print(top_annual)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
