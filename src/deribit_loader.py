"""Cliente de la API pública de Deribit (sin API key).

Endpoints usados:
  - get_volatility_index_data: histórico del DVOL (IV 30d por currency)
  - get_historical_volatility: vol realizada (Deribit la calcula internamente)
  - get_book_summary_by_currency: snapshot del option chain (IV, precios)

Uso:
    python -m src.deribit_loader
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import polars as pl
import requests

DERIBIT_BASE = "https://www.deribit.com/api/v2"
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

Currency = Literal["BTC", "ETH"]
Resolution = Literal["60", "3600", "86400"]  # 1m, 1h, 1d


def _dvol_cache_path(currency: Currency, resolution: Resolution) -> Path:
    return CACHE_DIR / f"DERIBIT_DVOL_{currency}_{resolution}.parquet"


def fetch_dvol(
    currency: Currency,
    start_ms: int,
    end_ms: int,
    resolution: Resolution = "86400",
    pause: float = 0.2,
) -> pl.DataFrame:
    """Descarga histórico del DVOL (índice de vol implícita 30d de Deribit).

    Devuelve DataFrame con: ts (datetime UTC), open, high, low, close (en puntos de vol %).
    """
    url = f"{DERIBIT_BASE}/public/get_volatility_index_data"
    frames: list[pl.DataFrame] = []
    cursor = start_ms
    # El endpoint devuelve hasta ~1000 puntos por request.
    for _ in range(50):
        params = {
            "currency": currency,
            "resolution": resolution,
            "start_timestamp": cursor,
            "end_timestamp": end_ms,
        }
        for attempt in range(5):
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        data = r.json().get("result", {})
        rows = data.get("data", [])
        if not rows:
            break
        df = pl.DataFrame(
            rows,
            schema=["ts_ms", "open", "high", "low", "close"],
            orient="row",
        )
        df = df.with_columns(
            pl.from_epoch("ts_ms", time_unit="ms").dt.replace_time_zone("UTC").alias("ts"),
        ).drop("ts_ms")
        frames.append(df)
        last_ts = int(rows[-1][0])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(pause)
    if not frames:
        return pl.DataFrame(schema={"ts": pl.Datetime("ms", "UTC"), "open": pl.Float64,
                                    "high": pl.Float64, "low": pl.Float64, "close": pl.Float64})
    return pl.concat(frames).unique(subset=["ts"]).sort("ts")


def load_dvol(
    currency: Currency,
    start: str,
    end: str,
    resolution: Resolution = "86400",
    refresh: bool = False,
) -> pl.DataFrame:
    """Carga DVOL con cache en parquet."""
    cache = _dvol_cache_path(currency, resolution)
    start_ts = pl.Series([start]).str.to_datetime(time_unit="ms", time_zone="UTC")[0]
    end_ts = pl.Series([end]).str.to_datetime(time_unit="ms", time_zone="UTC")[0]
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    if cache.exists() and not refresh:
        cached = pl.read_parquet(cache)
        bounds = cached.select(
            (pl.col("ts").min().dt.timestamp(time_unit="ms")).alias("s"),
            (pl.col("ts").max().dt.timestamp(time_unit="ms")).alias("e"),
        ).row(0)
        if bounds[0] <= start_ms and bounds[1] >= end_ms - 1:
            return cached.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts)).sort("ts")
    df = fetch_dvol(currency, start_ms, end_ms, resolution=resolution)
    if df.is_empty():
        raise RuntimeError(f"sin DVOL para {currency}")
    if cache.exists():
        old = pl.read_parquet(cache)
        df = pl.concat([old, df]).unique(subset=["ts"]).sort("ts")
    df.write_parquet(cache)
    return df.filter((pl.col("ts") >= start_ts) & (pl.col("ts") <= end_ts)).sort("ts")


if __name__ == "__main__":
    import sys
    cur = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    df = load_dvol(cur, "2022-01-01", "2026-07-04", resolution="86400")
    print(df.head())
    print(df.tail())
    print(f"filas: {df.height}")
