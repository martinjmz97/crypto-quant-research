"""Descarga masiva de klines 1h para todo el universo.

Uso:
    python -m src.download_universe
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from .data_loader import load_klines
from .universe import UNIVERSE

START = "2023-01-01"
END = "2024-12-31"
TIMEFRAME = "1h"


def main() -> int:
    print(f"Descargando {len(UNIVERSE)} símbolos, {TIMEFRAME}, {START}→{END}")
    print("-" * 60)
    t0 = time.time()
    for i, sym in enumerate(UNIVERSE, 1):
        ts = time.time()
        try:
            df = load_klines(sym, TIMEFRAME, START, END, market="perp", refresh=False)
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym:10s} {df.height:6d} velas  "
                  f"{df['open_time'].min()} → {df['open_time'].max()}  "
                  f"({time.time()-ts:.1f}s)")
        except Exception as e:
            print(f"[{i:2d}/{len(UNIVERSE)}] {sym:10s} ERROR: {e}")
    print("-" * 60)
    print(f"Total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.exit(main())
