"""Screen de cointegración sobre el universo de perps.

Para cada par (i, j), i < j:
  1. Alínea closes por open_time.
  2. Calcula log-precios.
  3. Engle-Granger: OLS + ADF sobre residuos.
  4. Reporta beta, pvalue, half-life, n.

Salida: strategies/cointegration/results/screen.csv con todos los pares,
y consolida top candidatos en stdout.

Uso:
    python -m strategies.cointegration.screen
"""
from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import polars as pl

from src.data_loader import _cache_path, load_klines
from src.stats import engle_granger
from src.universe import UNIVERSE

TIMEFRAME = "1h"
START = "2023-01-01"
END = "2024-12-31"

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = RESULTS_DIR / "screen_full.csv"


def load_close_matrix(symbols: list[str]) -> pl.DataFrame:
    """Carga closes de todos los símbolos alineados por open_time.

    Usa solo símbolos con cache local ya descargado.
    """
    frames: list[pl.DataFrame] = []
    for sym in symbols:
        cache = _cache_path(sym, TIMEFRAME, "perp")
        if not cache.exists():
            print(f"  [skip] {sym} sin cache")
            continue
        df = pl.read_parquet(cache).select(
            pl.col("open_time"),
            pl.col("close").alias(sym),
        )
        frames.append(df)
    # Join por open_time (inner: solo timestamps comunes a todos).
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, on="open_time", how="inner")
    return out.sort("open_time")


def screen_pairs(close_df: pl.DataFrame, pvalue_threshold: float = 0.05) -> pl.DataFrame:
    """Corre Engle-Granger sobre todos los pares."""
    symbols = [c for c in close_df.columns if c != "open_time"]
    pairs = list(itertools.combinations(symbols, 2))
    print(f"Testeando {len(pairs)} pares sobre {close_df.height} observaciones...")

    rows = []
    t0 = time.time()
    for k, (a, b) in enumerate(pairs, 1):
        # Log-precios; descartar NaNs o ceros.
        sub = close_df.select(["open_time", a, b]).drop_nulls()
        arr_a = np.log(sub[a].to_numpy())
        arr_b = np.log(sub[b].to_numpy())
        res = engle_granger(arr_a, arr_b)
        rows.append({
            "sym_y": a,
            "sym_x": b,
            "beta": round(res.beta, 6),
            "alpha": round(res.alpha, 6),
            "adf_stat": round(res.adf_stat, 4),
            "pvalue": res.pvalue,
            "half_life": round(res.half_life, 2),
            "n": res.n,
        })
        if k % 20 == 0 or k == len(pairs):
            elapsed = time.time() - t0
            print(f"  [{k}/{len(pairs)}] {elapsed:.1f}s  último: {a}/{b} p={res.pvalue:.4g}")

    df = pl.DataFrame(rows).sort("pvalue")
    return df


def main() -> int:
    print(f"=== Screen cointegración {START} → {END} ({TIMEFRAME}) ===")
    close_df = load_close_matrix(UNIVERSE)
    print(f"Matriz de closes: {close_df.height} filas, {len(close_df.columns)-1} símbolos")
    print(f"Rango: {close_df['open_time'].min()} → {close_df['open_time'].max()}")

    df = screen_pairs(close_df)
    df.write_csv(OUT_CSV)
    print(f"\nScreen completo: {OUT_CSV}")

    # Top candidatos: pvalue<0.05 y half-life jugable para hold de días (10-60 días = 240-1500h).
    top = df.filter(
        (pl.col("pvalue") < 0.05)
        & (pl.col("half_life").is_between(240.0, 1500.0))
        & (pl.col("half_life").is_not_nan())
    )
    print(f"\n=== Top {top.height} pares cointegrados (p<0.05, HL en [240h, 1500h] = 10-60 días) ===")
    with pl.Config(tbl_rows=30, tbl_width_chars=140, float_precision=4):
        print(top)
    top.write_csv(RESULTS_DIR / "screen_top.csv")
    print(f"\nTop guardado: {RESULTS_DIR}/screen_top.csv")
    return 0


if __name__ == "__main__":
    import sys
    from pathlib import Path as P
    sys.path.insert(0, str(P(__file__).resolve().parents[2]))
    sys.exit(main())
