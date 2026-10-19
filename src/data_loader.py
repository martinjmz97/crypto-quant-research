"""Descarga y cache de klines de Binance (API pública, sin API key).

Usa polars para almacenamiento y procesamiento. El cache se guarda en parquet.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import polars as pl
import requests

BINANCE_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
SPOT_KLINES_URL = "https://api.binance.com/api/v3/klines"
FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"

Timeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d"]
Market = Literal["perp", "spot"]

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Esquema canónico de klines. open_time es la clave única.
SCHEMA = pl.Schema({
    "open_time": pl.Datetime("ms", time_zone="UTC"),
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Datetime("ms", time_zone="UTC"),
})


def _cache_path(symbol: str, timeframe: Timeframe, market: Market) -> Path:
    suffix = "perp" if market == "perp" else "spot"
    return CACHE_DIR / f"{symbol}_{timeframe}_{suffix}.parquet"


def _raw_rows_to_lf(rows: list[list]) -> pl.DataFrame:
    """Convierte respuesta cruda de Binance en DataFrame tipado."""
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    df = pl.DataFrame(
        rows,
        schema={
            "open_time": pl.Int64, "open": pl.Utf8, "high": pl.Utf8,
            "low": pl.Utf8, "close": pl.Utf8, "volume": pl.Utf8,
            "close_time": pl.Int64, "_q": pl.Utf8, "_n": pl.Int64,
            "_tb": pl.Utf8, "_tq": pl.Utf8, "_ig": pl.Utf8,
        },
        orient="row",
    )
    df = (
        df.select(
            pl.from_epoch("open_time", time_unit="ms").dt.replace_time_zone("UTC"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
            pl.from_epoch("close_time", time_unit="ms").dt.replace_time_zone("UTC"),
        )
        .unique(subset=["open_time"])
        .sort("open_time")
    )
    return df


def fetch_klines(
    symbol: str,
    timeframe: Timeframe,
    start_ms: int,
    end_ms: int,
    market: Market = "perp",
    limit: int = 1500,
    pause: float = 0.15,
) -> pl.DataFrame:
    """Descarga klines OHLCV en ventana [start_ms, end_ms) por bloques."""
    url = BINANCE_KLINES_URL if market == "perp" else SPOT_KLINES_URL
    frames: list[pl.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": timeframe,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        for attempt in range(5):
            r = requests.get(url, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        batch = r.json()
        if not batch:
            break
        frames.append(_raw_rows_to_lf(batch))
        last_close_time = int(batch[-1][6])
        if last_close_time <= cursor:
            break
        cursor = last_close_time + 1
        time.sleep(pause)
    if not frames:
        return pl.DataFrame(schema=SCHEMA)
    return pl.concat(frames).unique(subset=["open_time"]).sort("open_time")


def load_klines(
    symbol: str,
    timeframe: Timeframe,
    start: str,
    end: str,
    market: Market = "perp",
    refresh: bool = False,
) -> pl.DataFrame:
    """Carga klines con cache en parquet. start/end en formato 'YYYY-MM-DD'."""
    cache = _cache_path(symbol, timeframe, market)
    # Conversión a ms epoch UTC.
    start_ts = pl.Series([start]).str.to_datetime(time_unit="ms", time_zone="UTC")[0]
    end_ts = pl.Series([end]).str.to_datetime(time_unit="ms", time_zone="UTC")[0]
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    if cache.exists() and not refresh:
        cached = pl.read_parquet(cache)
        # Comparar rangos en ms epoch usando polars expressions (evita extraer scalares).
        bounds = cached.select(
            (pl.col("open_time").min().dt.timestamp(time_unit="ms")).alias("have_start"),
            (pl.col("open_time").max().dt.timestamp(time_unit="ms")).alias("have_end"),
        ).row(0)
        have_start_ms, have_end_ms = bounds
        if have_start_ms <= start_ms and have_end_ms >= end_ms - 1:
            return (
                cached
                .filter((pl.col("open_time") >= start_ts) & (pl.col("open_time") <= end_ts))
                .sort("open_time")
            )
    df = fetch_klines(symbol, timeframe, start_ms, end_ms, market=market)
    if df.is_empty():
        raise RuntimeError(f"sin datos para {symbol} {timeframe} {market}")
    if cache.exists():
        old = pl.read_parquet(cache)
        df = pl.concat([old, df]).unique(subset=["open_time"]).sort("open_time")
    df.write_parquet(cache)
    return (
        df.filter((pl.col("open_time") >= start_ts) & (pl.col("open_time") <= end_ts))
        .sort("open_time")
    )


# --- Funding rate -----------------------------------------------------------

FUNDING_SCHEMA = pl.Schema({
    "funding_time": pl.Datetime("ms", time_zone="UTC"),
    "symbol": pl.Utf8,
    "funding_rate": pl.Float64,
    "mark_price": pl.Float64,
})


def _funding_cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}_funding.parquet"


def fetch_funding(
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    pause: float = 0.15,
) -> pl.DataFrame:
    """Descarga historial de funding rate. Endpoint público, sin API key."""
    frames: list[pl.DataFrame] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": limit,
        }
        for attempt in range(5):
            r = requests.get(FUNDING_URL, params=params, timeout=20)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            break
        batch = r.json()
        if not batch:
            break
        df = pl.DataFrame(batch).select(
            pl.from_epoch("fundingTime", time_unit="ms").dt.replace_time_zone("UTC").alias("funding_time"),
            pl.col("symbol"),
            pl.col("fundingRate").cast(pl.Float64).alias("funding_rate"),
            pl.when(pl.col("markPrice") == "")
            .then(None)
            .otherwise(pl.col("markPrice"))
            .cast(pl.Float64)
            .alias("mark_price"),
        )
        frames.append(df)
        last_t = int(batch[-1]["fundingTime"])
        if last_t <= cursor:
            break
        cursor = last_t + 1
        time.sleep(pause)
    if not frames:
        return pl.DataFrame(schema=FUNDING_SCHEMA)
    return pl.concat(frames).unique(subset=["funding_time"]).sort("funding_time")


def load_funding(
    symbol: str,
    start: str,
    end: str,
    refresh: bool = False,
) -> pl.DataFrame:
    """Carga funding rate con cache en parquet."""
    cache = _funding_cache_path(symbol)
    start_ts = pl.Series([start]).str.to_datetime(time_unit="ms", time_zone="UTC")[0]
    end_ts = pl.Series([end]).str.to_datetime(time_unit="ms", time_zone="UTC")[0]
    start_ms = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)

    if cache.exists() and not refresh:
        cached = pl.read_parquet(cache)
        bounds = cached.select(
            (pl.col("funding_time").min().dt.timestamp(time_unit="ms")).alias("s"),
            (pl.col("funding_time").max().dt.timestamp(time_unit="ms")).alias("e"),
        ).row(0)
        if bounds[0] <= start_ms and bounds[1] >= end_ms - 1:
            return (
                cached
                .filter((pl.col("funding_time") >= start_ts) & (pl.col("funding_time") <= end_ts))
                .sort("funding_time")
            )
    df = fetch_funding(symbol, start_ms, end_ms)
    if df.is_empty():
        raise RuntimeError(f"sin funding para {symbol}")
    if cache.exists():
        old = pl.read_parquet(cache)
        df = pl.concat([old, df]).unique(subset=["funding_time"]).sort("funding_time")
    df.write_parquet(cache)
    return (
        df.filter((pl.col("funding_time") >= start_ts) & (pl.col("funding_time") <= end_ts))
        .sort("funding_time")
    )


if __name__ == "__main__":
    import sys
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    tf = sys.argv[2] if len(sys.argv) > 2 else "1h"
    start = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
    end = sys.argv[4] if len(sys.argv) > 4 else "2024-06-01"
    df = load_klines(sym, tf, start, end)
    print(df.head())
    print(df.tail())
    print(f"filas: {df.height}")
