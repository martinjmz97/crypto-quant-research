"""EDA del Vol Risk Premium (VRP) = DVOL (IV 30d) - RV 30d.

Si VRP > 0 persistentemente, los compradores de opciones pagan más de lo que
la vol realizada justifica → short vol tiene edge.

Analizamos:
  - Distribución del VRP (media, mediana, percentiles)
  - Persistencia (autocorrelación)
  - Régimen (VRP positivo vs negativo)
  - P&L proxy: short straddle ATM cuando VRP > umbral, salir cuando VRP < 0

Uso:
    python -m strategies.derivatives.eda_vrp
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from src.deribit_loader import load_dvol
from strategies.derivatives.rv_calculator import compute_rv

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2023-10-09"
END = "2026-07-04"


def load_vrp(currency: str, underlying_symbol: str) -> pl.DataFrame:
    """Combina DVOL y RV en un DataFrame diario alineado."""
    dvol = load_dvol(currency, START, END, resolution="86400")
    dvol = dvol.select(
        pl.col("ts").dt.date().alias("date"),
        pl.col("close").alias("dvol"),
    )
    rv = compute_rv(underlying_symbol, window_hours=720)
    df = dvol.join(rv, on="date", how="inner").sort("date")
    df = df.with_columns(
        (pl.col("dvol") - pl.col("rv_close")).alias("vrp"),
    )
    return df


def ac1(arr: np.ndarray) -> float:
    a = arr[~np.isnan(arr)]
    if len(a) < 3:
        return float("nan")
    return float(np.corrcoef(a[:-1], a[1:])[0, 1])


def main() -> int:
    for currency, underlying in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")]:
        df = load_vrp(currency, underlying)
        vrp = df["vrp"].to_numpy()
        print(f"=== {currency} VRP ({df.height} días) ===")
        print(f"  DVOL  mean={df['dvol'].mean():.2f}  range [{df['dvol'].min():.2f}-{df['dvol'].max():.2f}]")
        print(f"  RV    mean={df['rv_close'].mean():.2f}  range [{df['rv_close'].min():.2f}-{df['rv_close'].max():.2f}]")
        print(f"  VRP   mean={np.nanmean(vrp):+.2f}  median={np.nanmedian(vrp):+.2f}")
        print(f"        std={np.nanstd(vrp):.2f}  AC1={ac1(vrp):.3f}")
        print(f"        % positivo: {(vrp > 0).mean()*100:.1f}%")
        print(f"        P5={np.nanpercentile(vrp,5):+.2f}  P25={np.nanpercentile(vrp,25):+.2f}  "
              f"P75={np.nanpercentile(vrp,75):+.2f}  P95={np.nanpercentile(vrp,95):+.2f}")
        # VRP por año.
        df_y = df.with_columns(pl.col("date").dt.year().alias("year"))
        for y in sorted(df_y["year"].unique()):
            sub = df_y.filter(pl.col("year") == y)
            v = sub["vrp"].to_numpy()
            print(f"    {y}: VRP mean={np.nanmean(v):+.2f}  %pos={np.nanmean(v>0)*100:.1f}%  n={sub.height}")
        df.write_csv(RESULTS_DIR / f"vrp_{currency}.csv")
        print()
    print(f"Guardado: {RESULTS_DIR}/vrp_BTC.csv y vrp_ETH.csv")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    sys.exit(main())
