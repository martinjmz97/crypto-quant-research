"""Descarga 1m y 5m para BTC + alts (universo lead-lag).

Uso:
    python -m src.download_lead_lag_data
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .data_loader import load_klines
from .universe_lead_lag import LEAD_LAG_UNIVERSE, LEADER

START = "2024-06-01"
END = "2024-12-31"
TIMEFRAMES = ["5m"]


def main() -> int:
    symbols = [LEADER] + LEAD_LAG_UNIVERSE
    print(f"=== Descarga lead-lag {START}→{END} ===")
    print(f"Símbolos: {len(symbols)}  Timeframes: {TIMEFRAMES}")
    t0 = time.time()
    for tf in TIMEFRAMES:
        print(f"\n--- Timeframe {tf} ---")
        for i, sym in enumerate(symbols, 1):
            ts = time.time()
            try:
                df = load_klines(sym, tf, START, END, market="perp", refresh=False)
                print(
                    f"  [{i:2d}/{len(symbols)}] {sym:10s} {df.height:7d} velas  "
                    f"{df['open_time'].min()} → {df['open_time'].max()}  "
                    f"({time.time()-ts:.1f}s)"
                )
            except Exception as e:
                print(f"  [{i:2d}/{len(symbols)}] {sym:10s} ERROR: {e}")
    print(f"\nTotal: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
